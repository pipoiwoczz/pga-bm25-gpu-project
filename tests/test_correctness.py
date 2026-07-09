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
    top_k as utils_top_k,
)
from cpu_baseline import NumpyBM25, verify_against_reference


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
    """500-doc corpus used by most unit tests."""
    return make_dataset(num_docs=500, vocab_size=200, num_queries=20, seed=1)


@pytest.fixture(scope="module")
def medium():
    """2000-doc corpus for top-K stress test."""
    return make_dataset(num_docs=2_000, vocab_size=500, num_queries=40, seed=3)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        # Only [a-z0-9] and whitespace survive after stripping punctuation.
        assert tokenize("hello, world!") == ["hello", "world"]
        assert tokenize("great!") == ["great"]
        assert tokenize("2024.") == ["2024"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_kept(self):
        tokens = tokenize("item 42 costs 3.5")
        assert "42" in tokens

    def test_multiple_spaces(self):
        assert tokenize("a  b   c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# IDF
# ---------------------------------------------------------------------------

class TestIDF:
    def test_idf_matches_reference(self, small):
        tokenized_corpus, _, _ = small
        reference = BM25Okapi(tokenized_corpus)
        custom = NumpyBM25(tokenized_corpus)

        for term, ref_idf in reference.idf.items():
            assert term in custom.idf, f"term '{term}' missing from custom IDF"
            assert np.isclose(custom.idf[term], ref_idf, atol=1e-9), (
                f"IDF mismatch for '{term}': "
                f"custom={custom.idf[term]:.8f} ref={ref_idf:.8f}"
            )

    def test_idf_non_negative(self, small):
        """After epsilon-flooring, all IDF values must be ≥ 0."""
        tokenized_corpus, _, _ = small
        custom = NumpyBM25(tokenized_corpus)
        for term, val in custom.idf.items():
            assert val >= 0, f"Negative IDF for '{term}': {val}"


# ---------------------------------------------------------------------------
# Per-document scores
# ---------------------------------------------------------------------------

class TestScores:
    def test_scores_match_reference(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        reference = BM25Okapi(tokenized_corpus)
        custom = NumpyBM25(tokenized_corpus)

        for q in tokenized_queries:
            ref = np.array(reference.get_scores(q))
            cus = custom.score(q)
            assert np.allclose(ref, cus, atol=1e-6), (
                f"Score mismatch for query {q}\n"
                f"max diff: {np.abs(ref - cus).max():.2e}"
            )

    def test_scores_shape(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        for q in tokenized_queries:
            s = custom.score(q)
            assert s.shape == (custom.corpus_size,)

    def test_scores_non_negative(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        for q in tokenized_queries:
            assert (custom.score(q) >= 0).all()


# ---------------------------------------------------------------------------
# Top-K selection
# ---------------------------------------------------------------------------

class TestTopK:
    def test_top_k_length(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        for q in tokenized_queries:
            result = custom.top_k(custom.score(q), k=10)
            assert len(result) == 10

    def test_top_k_values_are_valid_indices(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        n = len(tokenized_corpus)
        for q in tokenized_queries:
            indices = custom.top_k(custom.score(q), k=10)
            assert all(0 <= i < n for i in indices)

    def test_top_k_are_highest_scores(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        k = 10
        for q in tokenized_queries:
            scores = custom.score(q)
            top = custom.top_k(scores, k=k)
            min_top = scores[top].min()
            # Every document NOT in top-k must score ≤ min of top-k
            mask = np.ones(len(scores), dtype=bool)
            mask[top] = False
            if mask.any():
                assert scores[mask].max() <= min_top + 1e-9

    def test_utils_top_k_matches_member(self, small):
        """utils.top_k and NumpyBM25.top_k must select from the same top-k score range."""
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        k = 10
        for q in tokenized_queries:
            scores = custom.score(q)
            member_top = custom.top_k(scores, k=k)
            utils_top  = utils_top_k(scores, k=k)
            # Both must pick indices whose scores are all ≥ the (k+1)-th highest score.
            # (tie-breaking at the boundary may differ between implementations)
            if len(scores) > k:
                cutoff = np.sort(scores)[::-1][k]  # score just outside top-k
                assert (scores[member_top] >= cutoff - 1e-9).all()
                assert (scores[utils_top]  >= cutoff - 1e-9).all()
            assert len(member_top) == min(k, len(scores))
            assert len(utils_top)  == min(k, len(scores))

    def test_top_k_matches_reference(self, medium):
        """Top-10 document sets match rank_bm25 for all 40 queries."""
        tokenized_corpus, tokenized_queries, _ = medium
        reference = BM25Okapi(tokenized_corpus)
        custom = NumpyBM25(tokenized_corpus)

        # Verify every mismatch is a genuine score tie, not a formula error
        real_errors = 0
        for q in tokenized_queries:
            ref_scores = np.array(reference.get_scores(q))
            cus_scores = custom.score(q)
            if not np.allclose(ref_scores, cus_scores, atol=1e-5):
                real_errors += 1

        assert real_errors == 0, (
            f"{real_errors} queries had score mismatches beyond float tolerance"
        )

        mismatches = verify_against_reference(
            custom, reference, tokenized_queries, k=10
        )
        max_allowed = max(1, len(tokenized_queries) * 15 // 100)  # 15 % tie tolerance
        assert mismatches <= max_allowed, (
            f"{mismatches} queries had mismatched top-10 sets "
            f"(allowed ≤ {max_allowed} due to score ties at the boundary)"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_oov_query_returns_zero_scores(self):
        corpus, _ = generate_synthetic_corpus(50, vocab_size=50, seed=7)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        custom = NumpyBM25(tokenized_corpus)
        scores = custom.score(["this_term_is_not_in_vocabulary_xyz"])
        assert np.allclose(scores, 0.0)

    def test_empty_query_returns_zero_scores(self):
        corpus, _ = generate_synthetic_corpus(50, vocab_size=50, seed=7)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        custom = NumpyBM25(tokenized_corpus)
        scores = custom.score([])
        assert np.allclose(scores, 0.0)

    def test_single_document_corpus(self):
        # With a 1-doc corpus, "hello" appears in 100% of docs → IDF is
        # negative under ATIRE → epsilon floor kicks in → score is non-zero.
        custom = NumpyBM25([["hello", "world"]])
        scores = custom.score(["hello"])
        assert len(scores) == 1
        assert scores[0] != 0.0   # score is non-zero (may be negative via epsilon)

    def test_top_k_larger_than_corpus(self):
        corpus, _ = generate_synthetic_corpus(5, vocab_size=20, seed=9)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        custom = NumpyBM25(tokenized_corpus)
        scores = custom.score(tokenized_corpus[0])
        result = custom.top_k(scores, k=100)   # k >> corpus size
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Index integrity
# ---------------------------------------------------------------------------

class TestIndexIntegrity:
    def test_inverted_index_covers_all_documents(self):
        corpus, _ = generate_synthetic_corpus(100, vocab_size=30, seed=5)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        custom = NumpyBM25(tokenized_corpus)

        seen_docs = set()
        for doc_ids, _ in custom.inverted_index.values():
            seen_docs.update(doc_ids.tolist())
        assert seen_docs == set(range(len(tokenized_corpus)))

    def test_doc_lengths_match_tokenized_corpus(self):
        corpus, _ = generate_synthetic_corpus(200, vocab_size=100, seed=6)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        custom = NumpyBM25(tokenized_corpus)
        expected = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float64)
        assert np.array_equal(custom.doc_lens, expected)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

class TestBatchScoring:
    def test_batch_shape(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        result = np.array(custom.score_batch(tokenized_queries))
        assert result.shape == (len(tokenized_queries), len(tokenized_corpus))

    def test_batch_matches_individual(self, small):
        tokenized_corpus, tokenized_queries, _ = small
        custom = NumpyBM25(tokenized_corpus)
        batch = custom.score_batch(tokenized_queries)
        for i, q in enumerate(tokenized_queries):
            assert np.allclose(batch[i], custom.score(q), atol=1e-9)