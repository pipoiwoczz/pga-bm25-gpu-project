"""
bench_gpu_v1.py — GPU V1 (block-per-term) vs CPU baseline.

Usage (Colab T4):
    python benchmarks/bench_gpu_v1.py --num-docs 500000 --num-queries 32
"""
import time
import numpy as np
from numba import cuda
from common import default_argparser, build_dataset, bench_scoring, spot_check_top10
from cpu_baseline import NumpyBM25
from gpu_v1 import CudaBM25V1, _bm25_posting_kernel

def bench_stages_v1(gpu, queries, reps=10):
    """Per-stage breakdown of one V1 query. The 'H2D zero-reset' stage is
    the PCIe cost that V2's device-side _zero_kernel eliminates."""


    q = max(queries, key=len)
    threads = gpu.threads_per_block
    scores_host = np.empty(gpu.corpus_size, dtype=np.float64)
    ev = [cuda.event() for _ in range(2)]

    prep_ms, h2d_ms, reset_ms, kern_ms, d2h_ms = [], [], [], [], []

    for _ in range(reps):
        cuda.synchronize()
        t0 = time.perf_counter()
        term_ids = gpu._query_term_ids(q)
        t1 = time.perf_counter()
        prep_ms.append((t1 - t0) * 1000)

        d_tids = cuda.to_device(term_ids)
        cuda.synchronize()
        t2 = time.perf_counter()
        h2d_ms.append((t2 - t1) * 1000)

        # the naive reset: full 4 MB host->device copy of zeros
        gpu._d_scores.copy_to_device(gpu._zero_host)
        cuda.synchronize()
        t3 = time.perf_counter()
        reset_ms.append((t3 - t2) * 1000)

        ev[0].record()
        _bm25_posting_kernel[term_ids.shape[0], threads](
            d_tids, gpu.d_term_starts, gpu.d_term_ends,
            gpu.d_posting_doc_ids, gpu.d_posting_tfs,
            gpu.d_doc_lens, gpu.d_idf,
            gpu.k1, gpu.b, gpu.avgdl, gpu._d_scores,
        )
        ev[1].record()
        ev[1].synchronize()
        kern_ms.append(cuda.event_elapsed_time(ev[0], ev[1]))

        t4 = time.perf_counter()
        gpu._d_scores.copy_to_host(scores_host)
        cuda.synchronize()
        d2h_ms.append((time.perf_counter() - t4) * 1000)

    stages = [
        ("host prep (term ids)",          np.median(prep_ms)),
        ("H2D  term ids",                 np.median(h2d_ms)),
        ("H2D  zero-reset (4 MB!)",       np.median(reset_ms)),
        ("kernel: scoring    [device]",   np.median(kern_ms)),
        ("D2H  score vector",             np.median(d2h_ms)),
    ]
    accounted = sum(v for _, v in stages)

    print(f"\n  V1 stage breakdown (heaviest query, "
          f"grid={term_ids.shape[0]} blocks, median of {reps}):")
    print(f"    {'stage':<32} {'ms':>8}   {'%':>6}")
    print(f"    {'-' * 32} {'-' * 8}   {'-' * 6}")
    for name, ms in stages:
        print(f"    {name:<32} {ms:>8.3f}   {ms / accounted * 100:>5.1f}%")
    print(f"    {'-' * 32} {'-' * 8}   {'-' * 6}")
    print(f"    {'accounted total':<32} {accounted:>8.3f}   100.0%")
    return dict(stages)

def main():
    if not cuda.is_available():
        raise SystemExit("[skip] CUDA not available")

    args = default_argparser().parse_args()
    tokenized_corpus, queries = build_dataset(args)

    print("\n[build] Building indexes ...")
    cpu = NumpyBM25(tokenized_corpus)
    v1 = CudaBM25V1(tokenized_corpus)

    print(f"\n[score] Scoring {args.num_queries} queries:")
    t_cpu = bench_scoring(cpu.score, queries, label="CPU")
    t_v1 = bench_scoring(v1.score, queries, label="GPU V1",
                         sync_fn=cuda.synchronize)

    print(f"\n  Speedup V1 vs CPU : {t_cpu / t_v1:6.2f}x")
    bench_stages_v1(v1, queries)
    spot_check_top10(cpu, v1.score, v1.top_k, queries)


if __name__ == "__main__":
    main()