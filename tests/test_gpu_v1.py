import os
import sys

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

# Allow running from repo root or from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    generate_synthetic_corpus,
    generate_synthetic_queries,
    tokenize,
)
from cpu_baseline import NumpyBM25, verify_against_reference

# ---------------------------------------------------------------------------
# Skip this entire module unless numba + a CUDA device are actually available.
# This mirrors how gpu_v1 is lazily imported in main.py: CPU-only machines
# ---------------------------------------------------------------------------
numba = pytest.importorskip("numba", reason="numba not installed")
cuda = pytest.importorskip("numba.cuda", reason="numba.cuda not installed")

if not cuda.is_available():
    pytest.skip("No CUDA device available — skipping GPU V1 tests",
                allow_module_level=True)

from gpu_v1 import CudaBM25V1  # noqa: E402  (import after skip check)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_dataset(num_docs=500, vocab_size=200, num_queries=20, seed=1):
    corpus, vocab = generate_synthetic_corpus(num_docs, vocab_size=vocab_size, seed=seed)
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    queries = generate_synthetic_queries(vocab, num_queries, seed=seed + 1)
    tokenized_queries = [tokenize(q) for q in queries]
    return tokenized_corpus, tokenized_queries, vocab


@pytest.fixture(scope="module")
def small():
    """500-doc corpus — fast enough to run every test on."""
    return make_dataset(num_docs=500, vocab_size=200, num_queries=20, seed=1)


@pytest.fixture(scope="module")
def medium():
    """2000-doc corpus for the top-K stress test (mirrors test_correctness.py)."""
    return make_dataset(num_docs=2_000, vocab_size=500, num_queries=40, seed=3)


@pytest.fixture(scope="module")
def small_gpu(small):
    tokenized_corpus, _, _ = small
    return CudaBM25V1(tokenized_corpus)


@pytest.fixture(scope="module")
def medium_gpu(medium):
    tokenized_corpus, _, _ = medium
    return CudaBM25V1(tokenized_corpus)


# ---------------------------------------------------------------------------
# IDF (should be byte-for-byte identical logic to NumpyBM25, just re-verified
# here in case gpu_v1's inline IDF computation ever drifts from cpu_baseline)
# ---------------------------------------------------------------------------

class TestIDF:
    def test_idf_matches_reference(self, small, small_gpu):
        tokenized_corpus, _, _ = small
        reference = BM25Okapi(tokenized_corpus)

        for term, ref_idf in reference.idf.items():
            assert term in small_gpu.idf, f"term '{term}' missing from GPU V1 IDF table"
            assert np.isclose(small_gpu.idf[term], ref_idf, atol=1e-9), (
                f"IDF mismatch for '{term}': "
                f"gpu={small_gpu.idf[term]:.8f} ref={ref_idf:.8f}"
            )


# ---------------------------------------------------------------------------
# Per-document scores
# ---------------------------------------------------------------------------

class TestScores:
    def test_scores_match_cpu_baseline(self, small, small_gpu):
        """GPU V1 must match the CPU inverted-index scorer exactly (same
        formula, same accumulation order up to floating-point tolerance)."""
        tokenized_corpus, tokenized_queries, _ = small
        cpu = NumpyBM25(tokenized_corpus)

        for q in tokenized_queries:
            cpu_scores = cpu.score(q)
            gpu_scores = small_gpu.score(q)
            assert np.allclose(cpu_scores, gpu_scores, atol=1e-6), (
                f"GPU V1 vs CPU baseline mismatch for query {q}\n"
                f"max diff: {np.abs(cpu_scores - gpu_scores).max():.2e}"
            )

    def test_scores_match_reference(self, small, small_gpu):
        """GPU V1 must also match the rank_bm25 oracle directly."""
        tokenized_corpus, tokenized_queries, _ = small
        reference = BM25Okapi(tokenized_corpus)

        for q in tokenized_queries:
            ref = np.array(reference.get_scores(q))
            gpu = small_gpu.score(q)
            assert np.allclose(ref, gpu, atol=1e-6), (
                f"Score mismatch for query {q}\n"
                f"max diff: {np.abs(ref - gpu).max():.2e}"
            )

    def test_scores_shape_and_dtype(self, small, small_gpu):
        tokenized_corpus, tokenized_queries, _ = small
        for q in tokenized_queries:
            s = small_gpu.score(q)
            assert s.shape == (small_gpu.corpus_size,)
            assert s.dtype == np.float64

    def test_scores_non_negative(self, small, small_gpu):
        _, tokenized_queries, _ = small
        for q in tokenized_queries:
            assert (small_gpu.score(q) >= 0).all()


# ---------------------------------------------------------------------------
# Top-K selection
# ---------------------------------------------------------------------------

class TestTopK:
    def test_top_k_length(self, small, small_gpu):
        _, tokenized_queries, _ = small
        for q in tokenized_queries:
            result = small_gpu.top_k(small_gpu.score(q), k=10)
            assert len(result) == 10

    def test_top_k_are_highest_scores(self, small, small_gpu):
        _, tokenized_queries, _ = small
        k = 10
        for q in tokenized_queries:
            scores = small_gpu.score(q)
            top = small_gpu.top_k(scores, k=k)
            min_top = scores[top].min()
            mask = np.ones(len(scores), dtype=bool)
            mask[top] = False
            if mask.any():
                assert scores[mask].max() <= min_top + 1e-9

    def test_top_k_matches_reference(self, medium, medium_gpu):
        """Top-10 document sets match rank_bm25 for all 40 queries
        (allowing a small tie-tolerance at the boundary, same as the CPU
        baseline test)."""
        tokenized_corpus, tokenized_queries, _ = medium
        reference = BM25Okapi(tokenized_corpus)

        real_errors = 0
        for q in tokenized_queries:
            ref_scores = np.array(reference.get_scores(q))
            gpu_scores = medium_gpu.score(q)
            if not np.allclose(ref_scores, gpu_scores, atol=1e-5):
                real_errors += 1
        assert real_errors == 0, (
            f"{real_errors} queries had score mismatches beyond float tolerance"
        )

        mismatches = verify_against_reference(
            medium_gpu, reference, tokenized_queries, k=10
        )
        max_allowed = max(1, len(tokenized_queries) * 15 // 100)  # 15% tie tolerance
        assert mismatches <= max_allowed, (
            f"{mismatches} queries had mismatched top-10 sets "
            f"(allowed <= {max_allowed} due to score ties at the boundary)"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_oov_query_returns_zero_scores(self, small_gpu):
        scores = small_gpu.score(["this_term_is_not_in_vocabulary_xyz"])
        assert np.allclose(scores, 0.0)

    def test_empty_query_returns_zero_scores(self, small_gpu):
        scores = small_gpu.score([])
        assert np.allclose(scores, 0.0)

    def test_mixed_oov_and_valid_terms(self, small, small_gpu):
        """A query with some OOV terms mixed in should score identically to
        the same query with the OOV terms stripped out (they contribute 0
        posting-list entries, i.e. zero blocks launched for them)."""
        _, tokenized_queries, _ = small
        q = tokenized_queries[0] + ["definitely_not_a_real_term_zzz"]
        q_clean = tokenized_queries[0]
        assert np.allclose(small_gpu.score(q), small_gpu.score(q_clean), atol=1e-9)

    def test_single_document_corpus(self):
        # 1-doc corpus -> "hello" has 100% document frequency -> negative
        # IDF under ATIRE -> epsilon floor kicks in -> score is non-zero.
        gpu = CudaBM25V1([["hello", "world"]])
        scores = gpu.score(["hello"])
        assert len(scores) == 1
        assert scores[0] != 0.0

    def test_top_k_larger_than_corpus(self):
        corpus, _ = generate_synthetic_corpus(5, vocab_size=20, seed=9)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        gpu = CudaBM25V1(tokenized_corpus)
        scores = gpu.score(tokenized_corpus[0])
        result = gpu.top_k(scores, k=100)  # k >> corpus size
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

class TestBatchScoring:
    def test_batch_shape(self, small, small_gpu):
        _, tokenized_queries, _ = small
        result = np.array(small_gpu.score_batch(tokenized_queries))
        assert result.shape == (len(tokenized_queries), small_gpu.corpus_size)

    def test_batch_matches_individual(self, small, small_gpu):
        _, tokenized_queries, _ = small
        batch = small_gpu.score_batch(tokenized_queries)
        for i, q in enumerate(tokenized_queries):
            assert np.allclose(batch[i], small_gpu.score(q), atol=1e-9)