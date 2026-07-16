import math
from collections import defaultdict
import numpy as np
from numba import cuda

DEFAULT_THREADS_PER_BLOCK = 128  # 128 | 256 | 512 | 1024 - based on GPU architecture and memory constraints

@cuda.jit
def _bm25_posting_kernel(
    term_ids,          # (T,) int64 — term_id for each query term (this query)
    term_starts,        # (V,) int64 — CSR start offset per term_id
    term_ends,          # (V,) int64 — CSR end offset per term_id
    posting_doc_ids,    # (P,) int64 — flattened posting lists (doc_id)
    posting_tfs,        # (P,) float64 — flattened posting lists (term freq)
    doc_lens,            # (N,) float64
    idf_values,          # (V,) float64 — idf per term_id
    k1, b, avgdl,        # scalars
    scores,               # (N,) float64 — output, must be pre-zeroed
):
    block_id = cuda.blockIdx.x   # which query term this block handles
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x

    if block_id >= term_ids.shape[0]:
        return

    term_id = term_ids[block_id]
    start = term_starts[term_id]
    end = term_ends[term_id]
    idf = idf_values[term_id]

    i = start + tid
    while i < end:
        doc_id = posting_doc_ids[i]
        tf = posting_tfs[i]
        dl = doc_lens[doc_id]
        denom = tf + k1 * (1.0 - b + b * dl / avgdl)
        contrib = idf * tf * (k1 + 1.0) / denom
        cuda.atomic.add(scores, doc_id, contrib)
        i += stride

class CudaBM25V1:
    def __init__(self, tokenized_corpus, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25,
                 threads_per_block: int = DEFAULT_THREADS_PER_BLOCK):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.threads_per_block = threads_per_block

        self.corpus_size = len(tokenized_corpus)
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float64) 
        self.avgdl = np.mean(self.doc_lens)


        inverted_index = defaultdict(lambda: ([], []))  # term -> (doc_ids, tfs)
        df = defaultdict(int)          
        for doc_id, doc in enumerate(tokenized_corpus):
            counts = defaultdict(int)
            for tok in doc:
                counts[tok] += 1
            for term, tf in counts.items():
                doc_ids, tfs = inverted_index[term]
                doc_ids.append(doc_id)
                tfs.append(tf)
                df[term] += 1

        self.vocab = list(inverted_index.keys())
        self.term_to_id = {t: i for i, t in enumerate(self.vocab)}
        vocab_size = len(self.vocab)

        # ---- IDF, exactly matching rank_bm25's ATIRE variant + epsilon floor ----
        idf_dict, neg_terms, idf_sum = {}, [], 0.0
        for term, freq in df.items():
            val = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            idf_dict[term] = val
            idf_sum += val
            if val < 0:
                neg_terms.append(term)
        avg_idf = idf_sum / max(len(idf_dict), 1)
        eps = self.epsilon * avg_idf
        for term in neg_terms:
            idf_dict[term] = eps
        self.idf = idf_dict  # kept for CPU-facing parity / debugging

        idf_array = np.zeros(vocab_size, dtype=np.float64)
        for term, tid in self.term_to_id.items():
            idf_array[tid] = idf_dict[term]

        # ---- Flatten inverted index into CSR postings arrays ----
        term_starts = np.zeros(vocab_size, dtype=np.int64)
        term_ends = np.zeros(vocab_size, dtype=np.int64)
        total_postings = sum(len(v[0]) for v in inverted_index.values())
        posting_doc_ids = np.empty(total_postings, dtype=np.int64)
        posting_tfs = np.empty(total_postings, dtype=np.float64)

        offset = 0
        for term, tid in self.term_to_id.items():
            doc_ids, tfs = inverted_index[term]
            n = len(doc_ids)
            term_starts[tid] = offset
            posting_doc_ids[offset:offset + n] = doc_ids
            posting_tfs[offset:offset + n] = tfs
            offset += n
            term_ends[tid] = offset

        # ---- Upload everything that is constant across queries, ONCE ----
        self.d_doc_lens = cuda.to_device(self.doc_lens)
        self.d_idf = cuda.to_device(idf_array)
        self.d_term_starts = cuda.to_device(term_starts)
        self.d_term_ends = cuda.to_device(term_ends)
        self.d_posting_doc_ids = cuda.to_device(posting_doc_ids)
        self.d_posting_tfs = cuda.to_device(posting_tfs)

        # Reusable device score buffer (naive: re-zeroed every query)
        self._d_scores = cuda.device_array(self.corpus_size, dtype=np.float64)
        self._zero_host = np.zeros(self.corpus_size, dtype=np.float64)



    def _query_term_ids(self, query_tokens):
        ids = [self.term_to_id[t] for t in query_tokens if t in self.term_to_id]
        return np.array(ids, dtype=np.int64)

    def score(self, query_tokens):
        term_ids_host = self._query_term_ids(query_tokens)
        if term_ids_host.size == 0:
            return np.zeros(self.corpus_size, dtype=np.float64)

        d_term_ids = cuda.to_device(term_ids_host)
        # naive reset — full H2D copy every query (fixed in V2/V3)
        self._d_scores.copy_to_device(self._zero_host)

        blocks = term_ids_host.shape[0]          # one block per query term
        threads = self.threads_per_block
        _bm25_posting_kernel[blocks, threads](
            d_term_ids, self.d_term_starts, self.d_term_ends,
            self.d_posting_doc_ids, self.d_posting_tfs,
            self.d_doc_lens, self.d_idf,
            self.k1, self.b, self.avgdl,
            self._d_scores,
        )
        cuda.synchronize()

        scores_host = np.empty(self.corpus_size, dtype=np.float64)
        self._d_scores.copy_to_host(scores_host)
        return scores_host

    def score_batch(self, tokenized_queries):
        return [self.score(q) for q in tokenized_queries]

    def top_k(self, scores, k=10):
        if k >= len(scores):
            return np.argsort(-scores)
        idx = np.argpartition(-scores, k)[:k]
        return idx[np.argsort(-scores[idx])]
