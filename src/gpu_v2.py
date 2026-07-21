"""
gpu_v2.py — GPU V2: entry-parallel mapping + device-side buffer reset (Numba CUDA)

Two changes vs V1:

1. Device-side score reset.
   V1 zeroed the score buffer with a full host->device copy of a zero
   array (4 MB over PCIe per query at 500K docs). V2 zeroes it with a
   tiny grid-stride memset kernel that never leaves the GPU.

2. Entry-parallel work mapping ("posting list -> grid").
   V1 mapped one thread BLOCK per query term, so a query with 8 terms
   launched only 8 blocks, and the block owning a common term (long
   Zipfian posting list) ran orders of magnitude longer than the block
   owning a rare term. V2 flattens ALL posting entries of the query
   into one virtual work array of length E = sum(len(postings(t))) and
   distributes those E entries evenly across the whole grid with a
   grid-stride loop. Each thread locates which term its entry belongs
   to via binary search over a per-query prefix-offset array (T+1
   elements, precomputed on host). Load balance is now perfect
   regardless of posting-list skew, and the grid size scales with E,
   saturating the GPU even for short queries.
"""

import math
from collections import defaultdict

import numpy as np
from numba import cuda

DEFAULT_THREADS_PER_BLOCK = 256
MAX_BLOCKS = 4096  # cap; grid-stride loop covers any remainder


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@cuda.jit
def _zero_kernel(arr):
    """Grid-stride memset: arr[:] = 0.0, entirely on device."""
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < arr.shape[0]:
        arr[i] = 0.0
        i += stride


@cuda.jit
def _bm25_entry_kernel(
    term_ids,           # (T,)   int64  — term_id per query term
    q_offsets,          # (T+1,) int64  — prefix sum of posting-list lengths
    term_starts,        # (V,)   int64  — CSR start offset per term_id
    posting_doc_ids,    # (P,)   int64  — flattened posting lists (doc_id)
    posting_tfs,        # (P,)   float64 — flattened posting lists (term freq)
    doc_lens,           # (N,)   float64
    idf_values,         # (V,)   float64
    k1, b, avgdl,       # scalars
    scores,             # (N,)   float64 — output, pre-zeroed on device
):
    """One virtual work item per posting ENTRY of the query.

    Global work index g in [0, E) where E = q_offsets[T].
    Binary search over q_offsets maps g -> (query term slot, local index
    within that term's posting list). Grid-stride loop over g gives
    perfect load balance across Zipfian posting-list lengths.
    """
    total = q_offsets[q_offsets.shape[0] - 1]

    g = cuda.grid(1)
    stride = cuda.gridsize(1)

    while g < total:
        # --- binary search: largest t such that q_offsets[t] <= g ---
        lo = 0
        hi = q_offsets.shape[0] - 1          # == T
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if q_offsets[mid] <= g:
                lo = mid
            else:
                hi = mid
        t = lo                                # query term slot
        local = g - q_offsets[t]              # index within this posting list

        term_id = term_ids[t]
        p = term_starts[term_id] + local      # index into flattened postings

        doc_id = posting_doc_ids[p]
        tf = posting_tfs[p]
        dl = doc_lens[doc_id]
        denom = tf + k1 * (1.0 - b + b * dl / avgdl)
        contrib = idf_values[term_id] * tf * (k1 + 1.0) / denom
        cuda.atomic.add(scores, doc_id, contrib)

        g += stride


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CudaBM25V2:
    """Drop-in replacement for NumpyBM25 / CudaBM25V1 (same public interface)."""

    def __init__(self, tokenized_corpus, k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25,
                 threads_per_block: int = DEFAULT_THREADS_PER_BLOCK):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.threads_per_block = threads_per_block

        self.corpus_size = len(tokenized_corpus)
        self.doc_lens = np.array([len(doc) for doc in tokenized_corpus],
                                 dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lens))

        # ---- Build inverted index on CPU (identical to V1) ----
        inverted_index = defaultdict(lambda: ([], []))
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

        # ---- IDF: rank_bm25's ATIRE variant + epsilon floor over ALL terms ----
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
        self.idf = idf_dict  # CPU-facing parity / debugging

        idf_array = np.zeros(vocab_size, dtype=np.float64)
        for term, tid in self.term_to_id.items():
            idf_array[tid] = idf_dict[term]

        # ---- Flatten inverted index into CSR postings arrays ----
        term_starts = np.zeros(vocab_size, dtype=np.int64)
        term_lengths = np.zeros(vocab_size, dtype=np.int64)
        total_postings = sum(len(v[0]) for v in inverted_index.values())
        posting_doc_ids = np.empty(total_postings, dtype=np.int64)
        posting_tfs = np.empty(total_postings, dtype=np.float64)

        offset = 0
        for term, tid in self.term_to_id.items():
            doc_ids, tfs = inverted_index[term]
            n = len(doc_ids)
            term_starts[tid] = offset
            term_lengths[tid] = n
            posting_doc_ids[offset:offset + n] = doc_ids
            posting_tfs[offset:offset + n] = tfs
            offset += n

        # Host copies kept for per-query prefix-offset computation
        self.term_starts = term_starts
        self.term_lengths = term_lengths

        # ---- Upload constant data to GPU, ONCE ----
        self.d_doc_lens = cuda.to_device(self.doc_lens)
        self.d_idf = cuda.to_device(idf_array)
        self.d_term_starts = cuda.to_device(term_starts)
        self.d_posting_doc_ids = cuda.to_device(posting_doc_ids)
        self.d_posting_tfs = cuda.to_device(posting_tfs)

        # Reusable device score buffer — reset ON DEVICE (change #1 vs V1)
        self._d_scores = cuda.device_array(self.corpus_size, dtype=np.float64)
        self._zero_blocks = min(
            (self.corpus_size + threads_per_block - 1) // threads_per_block,
            MAX_BLOCKS,
        )

    # ------------------------------------------------------------------
    def _query_term_ids(self, query_tokens):
        ids = [self.term_to_id[t] for t in query_tokens if t in self.term_to_id]
        return np.array(ids, dtype=np.int64)

    def score(self, query_tokens):
        term_ids_host = self._query_term_ids(query_tokens)
        if term_ids_host.size == 0:
            return np.zeros(self.corpus_size, dtype=np.float64)

        # Per-query prefix offsets over posting-list lengths (host, tiny)
        lengths = self.term_lengths[term_ids_host]           # (T,)
        q_offsets_host = np.zeros(term_ids_host.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=q_offsets_host[1:])
        total_entries = int(q_offsets_host[-1])
        if total_entries == 0:
            return np.zeros(self.corpus_size, dtype=np.float64)

        d_term_ids = cuda.to_device(term_ids_host)           # T int64  (~64 B)
        d_q_offsets = cuda.to_device(q_offsets_host)         # T+1 int64

        threads = self.threads_per_block

        # Change #1: device-side reset — no PCIe traffic
        _zero_kernel[self._zero_blocks, threads](self._d_scores)

        # Change #2: grid sized by total posting ENTRIES, not by #terms
        blocks = min((total_entries + threads - 1) // threads, MAX_BLOCKS)
        _bm25_entry_kernel[blocks, threads](
            d_term_ids, d_q_offsets,
            self.d_term_starts,
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