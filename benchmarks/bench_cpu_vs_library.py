import numpy as np
from rank_bm25 import BM25Okapi

from common import (
    default_argparser, build_dataset, bench_scoring,
    time_index_build,
)
from cpu_baseline import NumpyBM25, verify_against_reference


def main():
    args = default_argparser(num_docs=100_000).parse_args()
    tokenized_corpus, queries = build_dataset(args)

    print("\n[build] Index construction (one-time, offline cost):")
    library, t_build_lib = time_index_build(
        lambda: BM25Okapi(tokenized_corpus), label="rank_bm25")
    custom, t_build_cus = time_index_build(
        lambda: NumpyBM25(tokenized_corpus), label="NumpyBM25")

    print(f"\n[score] Scoring {args.num_queries} queries "
          f"(per-query, amortized cost):")
    t_lib = bench_scoring(library.get_scores, queries, label="rank_bm25")
    t_cus = bench_scoring(custom.score, queries, label="NumpyBM25")

    print(f"\n  NumpyBM25 vs rank_bm25 (scoring): {t_lib / t_cus:6.2f}x")

    # Full correctness check — every query's top-10 must agree
    mismatches = verify_against_reference(custom, library, queries, k=10)
    print(f"  Correctness: {len(queries) - mismatches}/{len(queries)} "
          f"top-10 matched")

    # Numbers for the report's amortized-cost framing
    per_query_ms = t_cus / len(queries) * 1000
    print(f"\n  [report] index build {t_build_cus:.1f} s is one-time/offline; "
          f"scoring is {per_query_ms:.3f} ms/query — the GPU target is "
          f"scoring only (see Amdahl/amortization argument in proposal).")


if __name__ == "__main__":
    main()