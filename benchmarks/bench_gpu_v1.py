"""
bench_gpu_v1.py — GPU V1 (block-per-term) vs CPU baseline.

Usage (Colab T4):
    python benchmarks/bench_gpu_v1.py --num-docs 500000 --num-queries 32
"""

from numba import cuda

from common import default_argparser, build_dataset, bench_scoring, spot_check_top10
from cpu_baseline import NumpyBM25
from gpu_v1 import CudaBM25V1


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
    spot_check_top10(cpu, v1.score, v1.top_k, queries)


if __name__ == "__main__":
    main()