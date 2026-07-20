import argparse
import os
import sys
import time

import numpy as np

# Make src/ importable from benchmarks/ regardless of CWD
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils import (  # noqa: E402
    generate_synthetic_corpus,
    generate_queries_from_corpus,
    load_ag_news_corpus,
    tokenize,
)


def default_argparser(**overrides):
    """Common CLI flags for every bench script."""
    p = argparse.ArgumentParser()
    p.add_argument("--num-docs", type=int,
                   default=overrides.get("num_docs", 100_000))
    p.add_argument("--num-queries", type=int,
                   default=overrides.get("num_queries", 32))
    p.add_argument("--vocab-size", type=int,
                   default=overrides.get("vocab_size", 5_000))
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "ag_news"])
    p.add_argument("--seed", type=int, default=42)
    return p


def build_dataset(args):
    """Return (tokenized_corpus, tokenized_queries).

    Queries are ALWAYS sampled df-weighted from the actual corpus so
    they hit the long Zipfian posting lists — uniform vocabulary
    sampling produces deceptively fast timings (see proposal §3.2).
    """
    if args.dataset == "synthetic":
        print(f"[data] synthetic Zipfian corpus: {args.num_docs:,} docs, "
              f"vocab={args.vocab_size:,}")
        corpus, _ = generate_synthetic_corpus(
            args.num_docs, vocab_size=args.vocab_size, seed=args.seed)
    else:
        print(f"[data] AG News: max {args.num_docs:,} docs")
        corpus = load_ag_news_corpus(max_docs=args.num_docs)

    tokenized_corpus = [tokenize(d) for d in corpus]
    queries = generate_queries_from_corpus(
        tokenized_corpus, args.num_queries, seed=args.seed + 1)
    tokenized_queries = [tokenize(q) for q in queries]
    return tokenized_corpus, tokenized_queries


def bench_scoring(score_fn, queries, warmup=2, label="", sync_fn=None):
    """Time `score_fn(q)` over all queries; returns elapsed seconds.

    warmup runs are excluded (mandatory for Numba JIT; harmless for CPU).
    sync_fn (e.g. cuda.synchronize) is called before starting and after
    finishing the timed region for device-accurate wall-clock numbers.
    """
    for q in queries[:warmup]:
        score_fn(q)
    if sync_fn:
        sync_fn()

    t0 = time.perf_counter()
    for q in queries:
        score_fn(q)
    if sync_fn:
        sync_fn()
    elapsed = time.perf_counter() - t0

    print(f"  {label:<16} {elapsed * 1000:>10.2f} ms total "
          f"| {elapsed / len(queries) * 1000:>8.3f} ms/query "
          f"| {len(queries) / elapsed:>8.1f} q/s")
    return elapsed


def time_index_build(builder, label=""):
    """Time an index-construction callable; returns (instance, seconds)."""
    t0 = time.perf_counter()
    instance = builder()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<16} index build: {elapsed:>8.3f} s")
    return instance, elapsed


def spot_check_top10(reference_scorer, other_score_fn, other_top_k,
                     queries, n=5, k=10):
    """Top-10 agreement between a NumpyBM25-style reference and another scorer."""
    ok = 0
    for q in queries[:n]:
        ref = set(reference_scorer.top_k(reference_scorer.score(q), k=k).tolist())
        got = set(other_top_k(other_score_fn(q), k=k).tolist())
        ok += (ref == got)
    print(f"\n  Correctness spot-check: {ok}/{n} top-{k} matched")
    return ok == n