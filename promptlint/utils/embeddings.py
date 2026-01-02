from __future__ import annotations

import math
from typing import Iterable, List


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def average_pairwise_cosine(vectors: Iterable[List[float]]) -> float:
    vecs = list(vectors)
    if len(vecs) < 2:
        return 1.0
    total = 0.0
    count = 0
    for idx in range(len(vecs)):
        for jdx in range(idx + 1, len(vecs)):
            total += cosine_similarity(vecs[idx], vecs[jdx])
            count += 1
    return total / count if count else 0.0
