"""
bench_gpu_v1.py — timing & bottleneck profiling for GPU V1 (CudaBM25V1)

Three things this file measures, each targeting one of the three known
V1 limitations documented in the proposal:

  1. time_scoring_gpu        -> overall wall-clock, with mandatory warmup
                                 (first kernel launch pays Numba JIT cost)
  2. profile_stage_breakdown -> splits one score() call into:
                                   H2D term_ids upload
                                   naive score-buffer reset (H2D zero copy)
                                   kernel execution
                                   D2H result copy
                                 using cuda.event for device-accurate timing
  3. profile_load_imbalance  -> per-block (per query term) posting-list
                                 length distribution -> quantifies the
                                 Zipfian load imbalance that V2 addresses

Run standalone:
    python src/bench_gpu_v1.py --num-docs 500000 --num-queries 32
"""

import argparse
import time
from typing import List

import numpy as np
from numba import cuda

from utils import tokenize, generate_synthetic_corpus, generate_synthetic_queries, timer
from cpu_baseline import NumpyBM25, time_scoring as time_scoring_cpu
from gpu_v1 import CudaBM25V1, _bm25_posting_kernel


# ---------------------------------------------------------------------------
# 1. Wall-clock timing (with warmup)
# ---------------------------------------------------------------------------

def time_scoring_gpu(scorer: CudaBM25V1, tokenized_queries: List[List[str]],
                      warmup: int = 2) -> float:
    """Wall-clock seconds to score all queries, one pass.

    warmup runs are executed first and discarded — Numba CUDA JIT-compiles
    the kernel on its first launch, which can take 1-2 s and would
    otherwise dominate the measurement for small query batches.
    """
    for i in range(min(warmup, len(tokenized_queries))):
        scorer.score(tokenized_queries[i])
    cuda.synchronize()

    t0 = time.perf_counter()
    for q in tokenized_queries:
        scorer.score(q)
    cuda.synchronize()  # make sure the last kernel has actually finished
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 2. Stage-level breakdown using cuda.event (device-side, not wall-clock)
# ---------------------------------------------------------------------------

def profile_stage_breakdown(scorer: CudaBM25V1, tokenized_queries: List[List[str]],
                             warmup: int = 2) -> dict:
    """Average per-query time (ms) spent in each stage of score(), measured
    with cuda.event pairs so JIT/launch overhead on the Python side doesn't
    pollute the numbers.

    Returns a dict with keys: h2d_term_ids, reset_buffer, kernel, d2h_result, total
    """
    # warmup (JIT compile the kernel once, outside the measured region)
    for i in range(min(warmup, len(tokenized_queries))):
        scorer.score(tokenized_queries[i])
    cuda.synchronize()

    stages = {"h2d_term_ids": [], "reset_buffer": [], "kernel": [], "d2h_result": []}

    e_start = cuda.event()
    e_after_h2d = cuda.event()
    e_after_reset = cuda.event()
    e_after_kernel = cuda.event()
    e_after_d2h = cuda.event()

    scores_host = np.empty(scorer.corpus_size, dtype=np.float64)

    for q in tokenized_queries:
        term_ids_host = scorer._query_term_ids(q)
        if term_ids_host.size == 0:
            continue  # OOV/empty queries skip all GPU stages, not representative

        e_start.record()
        d_term_ids = cuda.to_device(term_ids_host)
        e_after_h2d.record()

        scorer._d_scores.copy_to_device(scorer._zero_host)
        e_after_reset.record()

        blocks = term_ids_host.shape[0]
        _bm25_posting_kernel[blocks, scorer.threads_per_block](
            d_term_ids, scorer.d_term_starts, scorer.d_term_ends,
            scorer.d_posting_doc_ids, scorer.d_posting_tfs,
            scorer.d_doc_lens, scorer.d_idf,
            scorer.k1, scorer.b, scorer.avgdl,
            scorer._d_scores,
        )
        e_after_kernel.record()

        scorer._d_scores.copy_to_host(scores_host)
        e_after_d2h.record()

        e_after_d2h.synchronize()  # wait for the whole chain before reading events

        stages["h2d_term_ids"].append(cuda.event_elapsed_time(e_start, e_after_h2d))
        stages["reset_buffer"].append(cuda.event_elapsed_time(e_after_h2d, e_after_reset))
        stages["kernel"].append(cuda.event_elapsed_time(e_after_reset, e_after_kernel))
        stages["d2h_result"].append(cuda.event_elapsed_time(e_after_kernel, e_after_d2h))

    summary = {k: float(np.mean(v)) for k, v in stages.items() if v}
    summary["total"] = sum(summary.values())
    return summary


# ---------------------------------------------------------------------------
# 3. Load-imbalance analysis across blocks (posting-list length per term)
# ---------------------------------------------------------------------------

def profile_load_imbalance(scorer: CudaBM25V1, tokenized_queries: List[List[str]]) -> dict:
    """For every query term actually launched as a block, record the
    posting-list length it must traverse. A wide spread (high max/mean
    ratio) is the direct, measurable symptom of Zipfian skew that V1's
    'one thread block per term' mapping does not compensate for — this is
    the evidence for why V2 (warp-level reduction / better work
    distribution) is needed.
    """
    lengths = []
    for q in tokenized_queries:
        for term in q:
            tid = scorer.term_to_id.get(term)
            if tid is None:
                continue
            length = int(scorer.d_term_ends[tid] - scorer.d_term_starts[tid]) \
                if False else None  # placeholder, replaced below
    # NOTE: term_starts/term_ends live on device; pull the host-side arrays
    # once instead of indexing the device array per term (avoids N tiny D2H copies).
    term_starts_host = scorer.d_term_starts.copy_to_host()
    term_ends_host = scorer.d_term_ends.copy_to_host()

    lengths = []
    n_blocks_per_query = []
    for q in tokenized_queries:
        term_ids = scorer._query_term_ids(q)
        n_blocks_per_query.append(len(term_ids))
        for tid in term_ids:
            lengths.append(int(term_ends_host[tid] - term_starts_host[tid]))

    lengths = np.array(lengths, dtype=np.int64)
    if lengths.size == 0:
        return {}

    return {
        "num_blocks_total": int(lengths.size),
        "avg_blocks_per_query": float(np.mean(n_blocks_per_query)),
        "posting_len_min": int(lengths.min()),
        "posting_len_max": int(lengths.max()),
        "posting_len_mean": float(lengths.mean()),
        "posting_len_std": float(lengths.std()),
        "imbalance_ratio_max_over_mean": float(lengths.max() / max(lengths.mean(), 1e-9)),
    }


# ---------------------------------------------------------------------------
# Full benchmark report (ties everything together, mirrors main.py's style)
# ---------------------------------------------------------------------------

def run_benchmark(num_docs=10_000, num_queries=32, vocab_size=5_000, seed=42):
    print(f"[1/4] Generating synthetic corpus ({num_docs:,} docs, vocab={vocab_size:,}) ...")
    with timer("corpus generation"):
        corpus, vocab = generate_synthetic_corpus(num_docs, vocab_size=vocab_size, seed=seed)
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    queries = generate_synthetic_queries(vocab, num_queries, seed=seed + 1)
    tokenized_queries = [tokenize(q) for q in queries]

    print("[2/4] Building CPU baseline + GPU V1 index ...")
    with timer("CPU index build"):
        cpu_scorer = NumpyBM25(tokenized_corpus)
    with timer("GPU V1 index build (incl. H2D upload)"):
        gpu_scorer = CudaBM25V1(tokenized_corpus)

    print("[3/4] Timing scoring step (CPU vs GPU V1, with warmup) ...")
    cpu_time = time_scoring_cpu(cpu_scorer, tokenized_queries)
    gpu_time = time_scoring_gpu(gpu_scorer, tokenized_queries, warmup=2)
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    print("[4/4] Stage breakdown + load-imbalance profiling (GPU V1) ...")
    breakdown = profile_stage_breakdown(gpu_scorer, tokenized_queries, warmup=2)
    imbalance = profile_load_imbalance(gpu_scorer, tokenized_queries)

    print()
    print("=" * 60)
    print("  GPU V1 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Corpus size:        {num_docs:>10,} documents")
    print(f"  Batch size:         {num_queries:>10,} queries")
    print(f"  CPU scoring time:   {cpu_time*1000:>10.2f} ms total "
          f"({cpu_time/num_queries*1000:.3f} ms/query)")
    print(f"  GPU V1 scoring time:{gpu_time*1000:>10.2f} ms total "
          f"({gpu_time/num_queries*1000:.3f} ms/query)")
    print(f"  Speedup (CPU/GPU):  {speedup:>10.2f}x")
    print("-" * 60)
    print("  Per-query stage breakdown (device-timed, avg over queries, ms):")
    for stage, ms in breakdown.items():
        tag = "  <-- naive full-buffer reset (V1-only cost)" if stage == "reset_buffer" else ""
        print(f"    {stage:<15s}: {ms:>8.4f} ms{tag}")
    print("-" * 60)
    print("  Load imbalance across blocks (posting-list length per term):")
    for k, v in imbalance.items():
        print(f"    {k:<28s}: {v:>10.2f}")
    print("=" * 60)

    return {
        "cpu_time": cpu_time,
        "gpu_time": gpu_time,
        "speedup": speedup,
        "breakdown": breakdown,
        "imbalance": imbalance,
    }


def parse_args():
    p = argparse.ArgumentParser(description="GPU V1 benchmark / bottleneck profiler")
    p.add_argument("--num-docs", type=int, default=10_000)
    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        num_docs=args.num_docs,
        num_queries=args.num_queries,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )