import argparse
import time
import numpy as np
from rank_bm25 import BM25Okapi

from utils import (generate_synthetic_corpus, generate_synthetic_queries, generate_queries_from_corpus,
                   load_ag_news_corpus, tokenize)
from cpu_baseline import NumpyBM25

def main():
    parser = argparse.ArgumentParser(description="BM25 CPU baseline")
    parser.add_argument("--num-docs", type=int, default=20000,
                         help="Corpus size (synthetic). Use 100_000 / "
                              "1_000_000 for the real benchmark runs.")
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--num-queries", type=int, default=32,
                         help="Matches the proposal's batch-of-32 target.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--profile", action="store_true",
                         help="Print a cProfile breakdown of the custom scorer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["synthetic", "ag_news"],
                         default="synthetic",
                         help="synthetic: Zipf-generated corpus (no internet "
                              "needed). ag_news: real ~120K-document news "
                              "corpus from Hugging Face (requires `pip "
                              "install datasets` and internet access).")
    args = parser.parse_args()

    if args.dataset == "ag_news":
        print(f"[1/5] Loading AG News corpus from Hugging Face "
              f"(max {args.num_docs} docs) ...")
        corpus = load_ag_news_corpus(max_docs=args.num_docs)
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        queries = generate_queries_from_corpus(
            tokenized_corpus, args.num_queries, seed=args.seed + 1)
        tokenized_queries = [tokenize(q) for q in queries]
        print(f"      Loaded {len(corpus)} real documents.")
    else:
        print(f"[1/5] Generating synthetic corpus: {args.num_docs} docs, "
              f"vocab={args.vocab_size} ...")
        corpus, vocab = generate_synthetic_corpus(
            args.num_docs, vocab_size=args.vocab_size, seed=args.seed)
        tokenized_corpus = [tokenize(doc) for doc in corpus]

        queries = generate_synthetic_queries(vocab, args.num_queries, seed=args.seed + 1)
        tokenized_queries = [tokenize(q) for q in queries]
    print(f"      {len(queries)} queries generated "
          f"(avg {np.mean([len(q) for q in tokenized_queries]):.1f} terms/query)")

    print("[2/5] Building rank_bm25 reference index ...")
    t0 = time.perf_counter()
    reference = BM25Okapi(tokenized_corpus)
    ref_build_time = time.perf_counter() - t0

    print("[3/5] Building custom NumPy inverted index ...")
    t0 = time.perf_counter()
    custom = NumpyBM25(tokenized_corpus)
    custom_build_time = time.perf_counter() - t0

    print("[4/5] Verifying correctness (top-{} match vs rank_bm25) ...".format(args.top_k))
    mismatches = 0
    for q in tokenized_queries:
        ref_scores = np.array(reference.get_scores(q))
        ref_top = set(np.argsort(-ref_scores)[:args.top_k].tolist())
        cus_scores = custom.score(q)
        cus_top = set(custom.top_k(cus_scores, k=args.top_k).tolist())
        if ref_top != cus_top:
            mismatches += 1
    print(f"      {len(tokenized_queries) - mismatches}/{len(tokenized_queries)} "
          f"queries had an exact top-{args.top_k} match.")
    if mismatches:
        print("      NOTE: mismatches are expected only from tie-breaking on "
              "equal scores; investigate if this number is large.")

    print("[5/5] Timing scoring step (this is the GPU target) ...")
    ref_time = time_reference_scoring(reference, tokenized_queries)
    custom_time = time_custom_scoring(custom, tokenized_queries)

    print("\n===== CPU BASELINE RESULTS =====")
    print(f"Corpus size:                 {args.num_docs:,} documents")
    print(f"Vocabulary size:              {args.vocab_size:,} terms")
    print(f"Batch size (queries):         {args.num_queries}")
    print(f"Index build time (rank_bm25): {ref_build_time:.4f} s")
    print(f"Index build time (custom):    {custom_build_time:.4f} s")
    print(f"Scoring time (rank_bm25):     {ref_time*1000:.2f} ms  "
          f"({ref_time/args.num_queries*1000:.3f} ms/query)")
    print(f"Scoring time (custom NumPy):  {custom_time*1000:.2f} ms  "
          f"({custom_time/args.num_queries*1000:.3f} ms/query)")
    print(f"Performance target (V3 GPU):  < 100 ms for 500K docs, 32 queries")
    print("=================================\n")

    if args.profile:
        print("Profiling custom scorer (top cumulative-time functions):\n")
        print(profile_custom_scoring(custom, tokenized_queries))


if __name__ == "__main__":
    main()