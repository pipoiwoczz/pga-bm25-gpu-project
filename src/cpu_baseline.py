import numpy as np
from collections import defaultdict

class NumpyBM25:
    """
    A from-scratch BM25 implementation built around an explicit inverted
    index (term -> posting list of (doc_id, term_freq)), matching the
    scoring formula used by rank_bm25.BM25Okapi so results are directly
    comparable.
    """

    def __init__(self, tokenized_corpus, k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = len(tokenized_corpus)
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus],
                                  dtype=np.float64)
        self.avgdl = self.doc_lens.mean()

        # Inverted index: term -> (doc_ids array, term_freq array)
        self.inverted_index = defaultdict(lambda: ([], []))
        df = defaultdict(int)  # document frequency per term

        for doc_id, doc in enumerate(tokenized_corpus):
            counts = defaultdict(int)
            for tok in doc:
                counts[tok] += 1
            for term, tf in counts.items():
                doc_ids, tfs = self.inverted_index[term]
                doc_ids.append(doc_id)
                tfs.append(tf)
                df[term] += 1

        # Freeze posting lists into NumPy arrays for fast vectorized access
        for term, (doc_ids, tfs) in self.inverted_index.items():
            self.inverted_index[term] = (
                np.array(doc_ids, dtype=np.int64),
                np.array(tfs, dtype=np.float64),
            )

        self.idf = self._compute_idf(df)

    def _compute_idf(self, df):
        """Mirrors rank_bm25.BM25Okapi's IDF computation exactly (the ATIRE
        BM25 variant: idf = log(N - freq + 0.5) - log(freq + 0.5), with an
        epsilon floor applied to any negative IDF values)."""
        idf = {}
        neg_idf_terms = []
        idf_sum = 0.0
        for term, freq in df.items():
            val = np.log(self.corpus_size - freq + 0.5) - np.log(freq + 0.5)
            idf[term] = val
            idf_sum += val
            if val < 0:
                neg_idf_terms.append(term)

        avg_idf = idf_sum / max(len(idf), 1)
        eps = self.epsilon * avg_idf
        for term in neg_idf_terms:
            idf[term] = eps
        return idf

    def score(self, query_tokens):
        """
        Scores ALL documents for a single query.
        This is the bottleneck: for every query term we traverse its
        posting list (irregular length!) and accumulate a weighted
        contribution into a dense score vector. On GPU this becomes
        one thread block per query term with atomic accumulation (V1),
        then warp reductions + shared memory (V2), then batched queries (V3).
        """
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf.get(term, 0.0)
            doc_ids, tfs = self.inverted_index[term]

            doc_lens = self.doc_lens[doc_ids]
            denom = tfs + self.k1 * (1 - self.b + self.b * doc_lens / self.avgdl)
            contrib = idf * (tfs * (self.k1 + 1)) / denom

            # Accumulation into the dense score vector (irregular scatter-add,
            # analogous to the atomicAdd step in the CUDA version)
            np.add.at(scores, doc_ids, contrib)
        return scores

    def score_batch(self, tokenized_queries):
        return [self.score(q) for q in tokenized_queries]

    def top_k(self, scores, k=10):
        if k >= len(scores):
            return np.argsort(-scores)
        idx = np.argpartition(-scores, k)[:k]
        return idx[np.argsort(-scores[idx])]