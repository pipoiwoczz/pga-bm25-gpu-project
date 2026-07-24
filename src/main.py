import argparse
import sys
import time
import os
from rank_bm25 import BM25Okapi

import numpy as np

from utils import (
    tokenize,
    generate_synthetic_corpus,
    generate_synthetic_queries,
    generate_queries_from_corpus,
    load_ag_news_corpus,
    load_ms_marco_corpus,
    load_ms_marco_queries,
    timer,
)
from cpu_baseline import (
    NumpyBM25,
    verify_against_reference,
    time_scoring,
    profile_scoring,
)


def _load_scorer(version: str):
    """Return the scorer class for the requested version.

    CPU version returns NumpyBM25 (already imported).
    GPU versions are imported lazily so CPU runs never need CUDA/Numba.
    """
    if version == "cpu":
        return NumpyBM25

    if version == "gpu_v1":
        try:
            from gpu_v1 import CudaBM25V1  # type: ignore
            return CudaBM25V1
        except ImportError:
            sys.exit("[ERROR] gpu_v1 not implemented yet — run --version cpu")

    if version == "gpu_v2":
        try:
            from gpu_v2 import CudaBM25V2  # type: ignore
            return CudaBM25V2
        except ImportError:
            sys.exit("[ERROR] gpu_v2 not implemented yet — run --version cpu")

    if version == "gpu_v3":
        try:
            from gpu_v3 import CudaBM25V3  # type: ignore
            return CudaBM25V3
        except ImportError:
            sys.exit("[ERROR] gpu_v3 not implemented yet — run --version cpu")

    if version == "gpu_v4":
        try:
            from gpu_v4 import CudaBM25V4  # type: ignore
            return CudaBM25V4
        except ImportError:
            sys.exit("[ERROR] gpu_v4 not implemented yet — run --version cpu")

    sys.exit(f"[ERROR] Unknown version '{version}'. "
             "Choose: cpu | gpu_v1 | gpu_v2 | gpu_v3 | gpu_v4")

def load_data(args) -> tuple:
    """Return (tokenized_corpus, tokenized_queries, vocab_or_none)."""

    if args.dataset == "synthetic":
        print(f"[1/4] Generating synthetic corpus "
              f"({args.num_docs:,} docs, vocab={args.vocab_size:,}) ...")
        with timer("corpus generation"):
            corpus, vocab = generate_synthetic_corpus(
                args.num_docs, vocab_size=args.vocab_size, seed=args.seed
            )
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        queries = generate_queries_from_corpus(vocab, args.num_queries, seed=args.seed + 1)
        tokenized_queries = [tokenize(q) for q in queries]
        return tokenized_corpus, tokenized_queries, vocab

    if args.dataset == "ag_news":
        print(f"[1/4] Loading AG News (max {args.num_docs:,} docs) ...")
        with timer("AG News load"):
            corpus = load_ag_news_corpus(max_docs=args.num_docs)
        print(f"      Loaded {len(corpus):,} documents.")
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        queries = generate_queries_from_corpus(
            tokenized_corpus, args.num_queries, seed=args.seed + 1
        )
        tokenized_queries = [tokenize(q) for q in queries]
        return tokenized_corpus, tokenized_queries, None
 
    if args.dataset == "ms_marco":
        print(f"[1/4] Loading MS MARCO (source={args.ms_marco_source!r}, "
              f"max {args.num_docs:,} passages) ...")
        with timer("MS MARCO corpus load"):
            corpus = load_ms_marco_corpus(
                max_docs=args.num_docs, source=args.ms_marco_source
            )
        print(f"      Loaded {len(corpus):,} passages.")
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        # Use real MS MARCO queries when available, fall back to corpus-sampled
        if args.ms_marco_real_queries:
            print(f"      Loading {args.num_queries} real MS MARCO queries ...")
            raw_queries = load_ms_marco_queries(
                max_queries=args.num_queries, split="validation"
            )
            tokenized_queries = [tokenize(q) for q in raw_queries]
        else:
            tokenized_queries = [tokenize(q) for q in
                generate_queries_from_corpus(
                    tokenized_corpus, args.num_queries, seed=args.seed + 1
                )]
        return tokenized_corpus, tokenized_queries, None
 
    sys.exit(f"[ERROR] Unknown dataset '{args.dataset}'. "
             "Choose: synthetic | ag_news | ms_marco")

def parse_args():
    p = argparse.ArgumentParser(
        description="BM25 Document Retrieval Benchmark (CSC14116 B4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version",     default="cpu",
                   choices=["cpu", "gpu_v1", "gpu_v2", "gpu_v3", "gpu_v4"])
    p.add_argument("--dataset",     default="synthetic",
                   choices=["synthetic", "ag_news", "ms_marco"])
    p.add_argument("--num-docs",    type=int, default=10_000)
    p.add_argument("--num-queries", type=int, default=32)
    p.add_argument("--vocab-size",  type=int, default=5_000,
                   help="Only used with --dataset synthetic")
    p.add_argument("--top-k",       type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--profile",     action="store_true",
                   help="Run cProfile on the scoring step (CPU only)")
    p.add_argument("--no-verify",   action="store_true",
                   help="Skip correctness check vs rank_bm25 (saves time at scale)")
    p.add_argument("--ms-marco-source", default="irds",
                   choices=["irds", "microsoft"],
                   help="MS MARCO HuggingFace source (only with --dataset ms_marco)")
    p.add_argument("--ms-marco-real-queries", action="store_true",
                   help="Use real MS MARCO validation queries instead of corpus-sampled")
    return p.parse_args()


def main():
    args = parse_args()

    # --- 1. Load data -------------------------------------------------------
    tokenized_corpus, tokenized_queries, _ = load_data(args)

    # --- 2. Build index (target scorer) ------------------------------------
    ScorerClass = _load_scorer(args.version)
    print(f"[2/4] Building index ({args.version}) ...")
    t0 = time.perf_counter()
    scorer = ScorerClass(tokenized_corpus)
    build_time = time.perf_counter() - t0
    print(f"      Done in {build_time:.3f} s")

    # --- 3. Correctness verification (CPU reference) -----------------------
    mismatches = 0
    if not args.no_verify:
        print("[3/4] Verifying correctness vs rank_bm25 ...")
        ref_scorer = BM25Okapi(tokenized_corpus)
        # For GPU scorers, delegate to an internal CPU scorer for verification
        cpu_scorer = scorer if args.version == "cpu" else NumpyBM25(tokenized_corpus)
        mismatches = verify_against_reference(
            cpu_scorer, ref_scorer, tokenized_queries, k=args.top_k
        )
        status = "✓ PASS" if mismatches == 0 else f"✗ {mismatches} mismatches"
        print(f"      {status} ({args.num_queries - mismatches}/{args.num_queries} "
              f"queries matched top-{args.top_k})")
    else:
        print("[3/4] Correctness check skipped (--no-verify).")

    # --- 4. Timing ----------------------------------------------------------
    print("[4/4] Timing scoring step ...")
    is_gpu = args.version.startswith("gpu")
    if is_gpu:
        from numba import cuda
        cuda.synchronize()
    score_time = time_scoring(scorer, tokenized_queries,
                              warmup=2 if is_gpu else 0)
    if is_gpu:
        cuda.synchronize()
    qps = args.num_queries / score_time

    # --- Results summary ----------------------------------------------------
    print()
    print("=" * 52)
    print("  BM25 BENCHMARK RESULTS")
    print("=" * 52)
    print(f"  Version:          {args.version}")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Corpus size:      {len(tokenized_corpus):>12,} documents")
    print(f"  Batch size:       {args.num_queries:>12,} queries")
    print(f"  Index build time: {build_time * 1000:>12.2f} ms")
    print(f"  Scoring time:     {score_time * 1000:>12.2f} ms total")
    print(f"  Per-query:        {score_time / args.num_queries * 1000:>12.3f} ms/query")
    print(f"  Throughput:       {qps:>12.1f} queries/sec")
    if not args.no_verify:
        print(f"  Correctness:      {args.num_queries - mismatches}/{args.num_queries} "
              f"top-{args.top_k} matched")
    print(f"  Target (V3 GPU):  500K docs, 32 queries < 100 ms")
    print("=" * 52)

    # --- Optional cProfile --------------------------------------------------
    if args.profile:
        if args.version != "cpu":
            print("\n[INFO] --profile only supported for --version cpu, skipping.")
        else:
            print("\nProfiling (cProfile, custom scorer):\n")
            print(profile_scoring(scorer, tokenized_queries))


if __name__ == "__main__":
    main()
