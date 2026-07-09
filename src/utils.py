import re
import numpy as np
import time
import contextlib
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from datasets import load_dataset

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text):
    """Tokenize a string into lowercase alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())

def generate_synthetic_corpus(num_docs, vocab_size=1000, seed=0):
    """Generate a synthetic corpus of documents."""
    rng = np.random.default_rng(seed)

    vocab = [f"term{i:05d}" for i in range(vocab_size)]
    # Zipf-distributed sampling weights over the vocabulary
    ranks = np.arange(1, vocab_size + 1)
    weights = 1.0 / ranks
    weights /= weights.sum()

    doc_lens = rng.integers(40, 200, size=num_docs) # document lengths between 40 and 200 words
    total_words = int(doc_lens.sum())

    # Single vectorized draw for the entire corpus, then split per document.
    word_indices = rng.choice(vocab_size, size=total_words, p=weights)
    words = np.asarray(vocab, dtype=object)[word_indices]

    offsets = np.concatenate(([0], np.cumsum(doc_lens)))
    corpus = [
        " ".join(words[offsets[i]:offsets[i + 1]])
        for i in range(num_docs)
    ]
    return corpus, vocab

def generate_synthetic_queries(vocab, num_queries: int, terms_per_query=(2, 5),
                                seed: int = 7):
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(num_queries):
        n_terms = rng.integers(terms_per_query[0], terms_per_query[1] + 1)
        q_terms = rng.choice(vocab, size=n_terms, replace=False)
        queries.append(" ".join(q_terms))
    return queries

def load_ag_news_corpus(max_docs: int = None):
    """
    Loads the AG News dataset from Hugging Face (~120,000 real news articles: title + description, 4 topics)
    """

    ds = load_dataset("fancyzhx/ag_news", split="train")
    corpus: List[str] = []
    for row in ds:
        corpus.append(row["text"])   # 'text' already contains title + body
        if max_docs and len(corpus) >= max_docs:
            break
    return corpus


def generate_queries_from_corpus(tokenized_corpus, num_queries: int,
                                  terms_per_query=(2, 5), seed: int = 7):
    """
    Builds realistic queries by sampling terms straight out of the corpus vocabulary
    """
    rng = np.random.default_rng(seed)
    all_terms = [tok for doc in tokenized_corpus for tok in doc]
    vocab = list(set(all_terms))
    queries = []
    for _ in range(num_queries):
        n_terms = rng.integers(terms_per_query[0], terms_per_query[1] + 1)
        idx = rng.choice(len(vocab), size=min(n_terms, len(vocab)), replace=False)
        q_terms = [vocab[i] for i in idx]
        queries.append(" ".join(q_terms))
    return queries

def load_ms_marco_corpus(
    max_docs: Optional[int] = None,
    source: str = "irds",
) -> List[str]:
    """
    Load MS MARCO passage corpus via HuggingFace datasets
    """
    corpus: List[str] = []
 
    if source == "irds":
        # Flat format: one passage per row — easiest to work with
        ds = load_dataset("irds/msmarco-passage", "docs", split="train", streaming=True)
        for row in ds:
            corpus.append(row["text"])
            if max_docs and len(corpus) >= max_docs:
                break
 
    elif source == "microsoft":
        # Nested format: each QA row contains multiple passages
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
        for row in ds:
            for passage in row["passages"]["passage_text"]:
                corpus.append(passage)
                if max_docs and len(corpus) >= max_docs:
                    return corpus
 
    else:
        raise ValueError(f"Unknown source {source!r}. Choose 'irds' or 'microsoft'.")
 
    return corpus
 
 
def load_ms_marco_queries(
    max_queries: Optional[int] = None,
    split: str = "validation",
) -> List[str]:
    """Load MS MARCO queries for benchmarking retrieval quality"""
    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split, streaming=True)
    queries: List[str] = []
    for row in ds:
        q = row.get("query", "").strip()
        if q:
            queries.append(q)
        if max_queries and len(queries) >= max_queries:
            break
    return queries
 

def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k highest scores (unordered within the k).

    Uses argpartition — O(n) instead of O(n log n) full sort.
    The GPU equivalent will use CUB DeviceRadixSort (V4) or a parallel
    reduction top-K kernel.
    """
    if k >= len(scores):
        return np.arange(len(scores))
    part = np.argpartition(scores, -k)[-k:]
    return part[np.argsort(-scores[part])]   # sort only the k winners

@contextlib.contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed wall-clock time.

    Usage::

        with timer("index build"):
            idx = NumpyBM25(corpus)
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if label:
        print(f"  [{label}] {elapsed * 1000:.2f} ms")