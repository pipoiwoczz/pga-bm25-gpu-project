import math
from collections import defaultdict
import numpy as np
from numba import cuda

DEFAULT_THREADS_PER_BLOCK = 128  # 128 | 256 | 512 | 1024 - based on GPU architecture and memory constraints

class CudaBM25V1:
    def __init__(self, tokenized_corpus, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25,
                 threads_per_block: int = DEFAULT_THREADS_PER_BLOCK):
        pass

    def _compute_idf(self, df):
        pass

    def score(self, query_tokens, doc_ids, doc_lens, avgdl, idf, k1, b):
        pass
