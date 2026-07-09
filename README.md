# PGA-BM25-GPU-Project

**CSC14116 – Applied Parallel Programming (Class 19_21)**
**Track B4 – BM25 Document Retrieval:** Accelerating Full-Corpus BM25 Scoring with Custom CUDA Kernels

| | |
|---|---|
| **Group** | ChillBoy |
| **Member** | Le Ngoc Anh Khoa – 22127196 |
| **Topic** | B4 – BM25 Document Retrieval |
| **Repo** | https://github.com/pipoiwoczz/pga-bm25-gpu-project |

## 1. Overview

BM25 is the default ranking function behind production search engines such as Elasticsearch, Solr, and Apache Lucene. Scoring a batch of queries against a full corpus requires traversing the posting list of every query term and accumulating a score for every matching document — a process that is serial on CPU but **embarrassingly parallel on GPU**, since every `(query term, document)` pair in a posting list contributes independently to the final score.

This project follows the **Partial GPU Principle**: index construction and top-K extraction stay on CPU (cheap, run once offline), and only the posting-list traversal + score accumulation step is replaced with a custom CUDA kernel — the step `cProfile` confirmed accounts for over 95% of runtime.

## 2. Project Structure

```
pga-bm25-gpu-project/
├── README.md
├── requirements.txt
├── src/
│   ├── cpu_baseline.py     # CPU baseline: rank_bm25 (oracle) + NumpyBM25 (timing)
│   ├── utils.py             # Tokenizer, inverted index builder, Zipfian corpus generator
│   ├── main.py               # Benchmark entry point (--num-docs, --num-queries, --profile)
│   └── gpu/                  # (upcoming) GPU kernels V1 → V4
├── tests/
│   └── test_correctness.py  # Verifies CPU top-10 matches rank_bm25 exactly (32/32 pass)
└── docs/
    └── proposal.pdf          # Project proposal
```

## 3. Current Status

- **CPU baseline complete** (`src/cpu_baseline.py`): includes `rank_bm25.BM25Okapi` as the correctness oracle and a hand-written `NumpyBM25` class (explicit inverted index) as the timing baseline. Both produce identical top-10 rankings on all 32 test queries.
- **Profiling** with `cProfile` confirms `NumpyBM25.score` (posting-list traversal + `np.add.at` scatter-add) as the bottleneck, satisfying the "Partial GPU Principle" requirement.
- **Project proposal** finished, typeset with Typst.
- **CUDA kernels V1–V4** (Numba CUDA / CuPy) — upcoming, see Optimization Plan below.

## 4. Dataset

| Source | Role | Size |
|---|---|---|
| AG News (`fancyzhx/ag_news`, HuggingFace) | Development / correctness validation | ~120,000 real news articles |
| MS MARCO Passage Ranking (`irds/msmarco-passage`) | Large-scale benchmark | up to 500,000 passages (streamed) |
| Synthetic Zipfian corpus (generated in `utils.py`) | GPU scaling tests, fallback if MS MARCO is unavailable | arbitrary size |

## 5. Setup

```bash
git clone https://github.com/pipoiwoczz/pga-bm25-gpu-project.git
cd pga-bm25-gpu-project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Usage

Run the CPU baseline with profiling:

```bash
python src/main.py --num-docs 500000 --num-queries 32 --profile
```

Run correctness tests:

```bash
pytest tests/test_correctness.py
```

*Note: You can go to [Notebook](https://colab.research.google.com/drive/14cCej1gV21cCvswxFGFLemGyu74BHUK-?usp=sharing) and run my notebook there*

## 7. CPU Baseline Results (500K documents, 32 queries)

```
Index build time: 47,855.81 ms
Dataset:           Synthetic Zipfian corpus (500,000 documents, 32 queries - 8 terms each)
Scoring time:      96.64 ms total (3.020 ms/query)
Throughput:        331.1 queries/sec
Correctness:       32/32 top-10 matched
```

GPU target (V3): score 500K documents for a 32-query batch in **under 100 ms**.

## 8. GPU Optimization Plan

| Stage | Technique | Addresses |
|---|---|---|
| V1 | One thread block per query term, naive atomicAdd | Functional GPU port |
| V2 | Warp-level reduction + shared-memory score buffer | Atomic contention |
| V3 | Batch all 32 queries simultaneously, shared inverted-index segments | GPU under-utilization |
| V4 (stretch) | `CUB DeviceRadixSort` for parallel top-K | Removes CPU round-trip for ranking |

## 9. References

- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Crane, M. et al. (2017). *Faster and Smaller Inverted Indices with Threading.* SIGIR 2017.
- Pibiri, G. & Trani, R. (2021). *Techniques for Inverted Index Compression.* ACM CSUR.
- [rank_bm25 (Dorian Brown)](https://github.com/dorianbrown/rank_bm25)
- [NVIDIA CUB library](https://nvlabs.github.io/cub/)
