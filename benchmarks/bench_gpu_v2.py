import numpy as np
from numba import cuda
import time
from common import default_argparser, build_dataset, bench_scoring, spot_check_top10
from cpu_baseline import NumpyBM25
from gpu_v1 import CudaBM25V1
from gpu_v2 import CudaBM25V2, _zero_kernel, _bm25_entry_kernel, MAX_BLOCKS


def bench_stages_v2(gpu, queries, reps=10):
    """Full per-stage breakdown of one V2 query: device kernels via
    cuda.event, host-side transfers via perf_counter.

    Device timings use the heaviest query (worst-case kernel work).
    Host timings are medians over `reps` runs of that same query, each
    stage isolated with cuda.synchronize() so no work overlaps the
    measurement window.
    """

    q = max(queries, key=len)
    threads = gpu.threads_per_block

    # ---- host-side prep (measured) ----
    prep_ms, h2d_ms, zero_ms, kern_ms, d2h_ms = [], [], [], [], []
    scores_host = np.empty(gpu.corpus_size, dtype=np.float64)

    ev = [cuda.event() for _ in range(3)]

    for _ in range(reps):
        # 1. term-id lookup + prefix offsets (pure Python/NumPy)
        cuda.synchronize()
        t0 = time.perf_counter()
        term_ids = gpu._query_term_ids(q)
        lengths = gpu.term_lengths[term_ids]
        q_offsets = np.zeros(term_ids.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=q_offsets[1:])
        t1 = time.perf_counter()
        prep_ms.append((t1 - t0) * 1000)

        # 2. H2D of the tiny per-query arrays
        d_tids = cuda.to_device(term_ids)
        d_offs = cuda.to_device(q_offsets)
        cuda.synchronize()
        t2 = time.perf_counter()
        h2d_ms.append((t2 - t1) * 1000)

        total = int(q_offsets[-1])
        blocks = min((total + threads - 1) // threads, MAX_BLOCKS)

        # 3. + 4. device kernels, timed with events
        ev[0].record()
        _zero_kernel[gpu._zero_blocks, threads](gpu._d_scores)
        ev[1].record()
        _bm25_entry_kernel[blocks, threads](
            d_tids, d_offs, gpu.d_term_starts,
            gpu.d_posting_doc_ids, gpu.d_posting_tfs,
            gpu.d_doc_lens, gpu.d_idf,
            gpu.k1, gpu.b, gpu.avgdl, gpu._d_scores,
        )
        ev[2].record()
        ev[2].synchronize()
        zero_ms.append(cuda.event_elapsed_time(ev[0], ev[1]))
        kern_ms.append(cuda.event_elapsed_time(ev[1], ev[2]))

        # 5. D2H of the full score vector
        t3 = time.perf_counter()
        gpu._d_scores.copy_to_host(scores_host)
        cuda.synchronize()
        d2h_ms.append((time.perf_counter() - t3) * 1000)

    stages = [
        ("host prep (term ids, offsets)", np.median(prep_ms)),
        ("H2D  per-query arrays",         np.median(h2d_ms)),
        ("kernel: zero-reset [device]",   np.median(zero_ms)),
        ("kernel: scoring    [device]",   np.median(kern_ms)),
        ("D2H  score vector",             np.median(d2h_ms)),
    ]
    accounted = sum(v for _, v in stages)

    total_entries = int(np.cumsum(gpu.term_lengths[gpu._query_term_ids(q)])[-1])
    print(f"\n  V2 stage breakdown (heaviest query, {total_entries:,} entries, "
          f"median of {reps}):")
    print(f"    {'stage':<32} {'ms':>8}   {'%':>6}")
    print(f"    {'-' * 32} {'-' * 8}   {'-' * 6}")
    for name, ms in stages:
        print(f"    {name:<32} {ms:>8.3f}   {ms / accounted * 100:>5.1f}%")
    print(f"    {'-' * 32} {'-' * 8}   {'-' * 6}")
    print(f"    {'accounted total':<32} {accounted:>8.3f}   100.0%")

    device_ms = np.median(zero_ms) + np.median(kern_ms)
    print(f"\n    device-only : {device_ms:.3f} ms  "
          f"({device_ms / accounted * 100:.0f}% of accounted)")
    print(f"    data movement + host : {accounted - device_ms:.3f} ms  "
          f"({(accounted - device_ms) / accounted * 100:.0f}%)")
    return dict(stages)


def main():
    if not cuda.is_available():
        raise SystemExit("[skip] CUDA not available")

    args = default_argparser().parse_args()
    tokenized_corpus, queries = build_dataset(args)

    print("\n[build] Building indexes (CPU / V1 / V2) ...")
    cpu = NumpyBM25(tokenized_corpus)
    v1 = CudaBM25V1(tokenized_corpus)
    v2 = CudaBM25V2(tokenized_corpus)

    print(f"\n[score] Scoring {args.num_queries} queries:")
    t_cpu = bench_scoring(cpu.score, queries, label="CPU")
    t_v1 = bench_scoring(v1.score, queries, label="GPU V1",
                         sync_fn=cuda.synchronize)
    t_v2 = bench_scoring(v2.score, queries, label="GPU V2",
                         sync_fn=cuda.synchronize)

    print(f"\n  Speedup V1 vs CPU : {t_cpu / t_v1:6.2f}x")
    print(f"  Speedup V2 vs CPU : {t_cpu / t_v2:6.2f}x")
    print(f"  Speedup V2 vs V1  : {t_v1 / t_v2:6.2f}x")

    bench_stages_v2(v2, queries)
    spot_check_top10(cpu, v2.score, v2.top_k, queries)


if __name__ == "__main__":
    main()