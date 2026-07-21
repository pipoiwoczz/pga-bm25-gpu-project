import os
import sys

import numpy as np
import pytest

numba = pytest.importorskip("numba")
from numba import cuda

if not cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import generate_synthetic_corpus, generate_synthetic_queries, tokenize
from cpu_baseline import NumpyBM25
from gpu_v2 import CudaBM25V2


def make_dataset(num_docs=800, vocab_size=300, num_queries=25, seed=11):
    corpus, vocab = generate_synthetic_corpus(num_docs, vocab_size=vocab_size, seed=seed)
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    queries = generate_synthetic_queries(vocab, num_queries, seed=seed + 1)
    tokenized_queries = [tokenize(q) for q in queries]
    return tokenized_corpus, tokenized_queries


@pytest.fixture(scope="module")
def data():
    return make_dataset()


@pytest.fixture(scope="module")
def scorers(data):
    tokenized_corpus, _ = data
    return NumpyBM25(tokenized_corpus), CudaBM25V2(tokenized_corpus)


class TestScores:
    def test_scores_match_cpu(self, data, scorers):
        _, queries = data
        cpu, gpu = scorers
        for q in queries:
            ref = cpu.score(q)
            got = gpu.score(q)
            assert np.allclose(ref, got, atol=1e-6), (
                f"max diff {np.abs(ref - got).max():.2e} for query {q}"
            )

    def test_top10_match_cpu(self, data, scorers):
        _, queries = data
        cpu, gpu = scorers
        for q in queries:
            ref_top = set(cpu.top_k(cpu.score(q), k=10).tolist())
            got_top = set(gpu.top_k(gpu.score(q), k=10).tolist())
            assert ref_top == got_top

    def test_repeated_queries_are_stable(self, data, scorers):
        """Guard against score-buffer reset bugs: same query twice in a row
        (and after a different query) must give identical results."""
        _, queries = data
        _, gpu = scorers
        q = queries[0]
        first = gpu.score(q)
        gpu.score(queries[1])          # pollute the buffer with another query
        again = gpu.score(q)
        assert np.array_equal(first, again)


class TestEdgeCases:
    def test_oov_query(self, scorers):
        _, gpu = scorers
        assert np.allclose(gpu.score(["zzz_not_in_vocab"]), 0.0)

    def test_empty_query(self, scorers):
        _, gpu = scorers
        assert np.allclose(gpu.score([]), 0.0)

    def test_single_term_query(self, data, scorers):
        tokenized_corpus, _ = data
        cpu, gpu = scorers
        term = tokenized_corpus[0][0]
        assert np.allclose(cpu.score([term]), gpu.score([term]), atol=1e-6)

    def test_duplicate_terms_in_query(self, data, scorers):
        """rank_bm25 semantics: duplicated query terms contribute twice."""
        tokenized_corpus, _ = data
        cpu, gpu = scorers
        term = tokenized_corpus[0][0]
        q = [term, term, term]
        assert np.allclose(cpu.score(q), gpu.score(q), atol=1e-6)


class TestBatch:
    def test_batch_matches_individual(self, data, scorers):
        _, queries = data
        _, gpu = scorers
        batch = gpu.score_batch(queries[:5])
        for i, q in enumerate(queries[:5]):
            assert np.allclose(batch[i], gpu.score(q), atol=1e-12)