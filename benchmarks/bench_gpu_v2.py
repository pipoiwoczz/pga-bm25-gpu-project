import numpy as np
from numba import cuda

from common import default_argparser, build_dataset, bench_scoring, spot_check_top10
from cpu_baseline import NumpyBM25
from gpu_v1 import CudaBM25V1
from gpu_v2 import CudaBM25V2, _zero_kernel, _bm25_entry_kernel, MAX_BLOCKS


def bench_v2_stages(gpu, queries, reps=10):
    """cuda.event timing of V2's two kernels on the heaviest query."""
    q = max(queries, key=len)
    term_ids = gpu._query_term_ids(q)
    lengths = gpu.term_lengths[term_ids]
    q_offsets = np.zeros(term_ids.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=q_offsets[1:])
    total = int(q_offsets[-1])
    threads = gpu.threads_per_block
    blocks = min((total + threads - 1) // threads, MAX_BLOCKS)

    d_tids = cuda.to_device(term_ids)
    d_offs = cuda.to_device(q_offsets)

    ev = [cuda.event() for _ in range(3)]
    zero_ms, kern_ms = [], []
    for _ in range(reps):
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

    print(f"\n  V2 stage timing (heaviest query, {total:,} entries, "
          f"{blocks} blocks x {threads} threads, median of {reps}):")
    print(f"    device zero-reset : {np.median(zero_ms):8.3f} ms")
    print(f"    scoring kernel    : {np.median(kern_ms):8.3f} ms")


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

    bench_v2_stages(v2, queries)
    spot_check_top10(cpu, v2.score, v2.top_k, queries)


if __name__ == "__main__":
    main()