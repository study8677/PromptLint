from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Dict

from promptlint.utils.text import non_empty_lines, tokenize


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def structural_signature(text: str) -> Dict[str, float]:
    lines = non_empty_lines(text)
    line_count = len(lines)
    bullet_count = sum(1 for line in lines if re.match(r"^\s*[-*]\s+", line))
    numbered_count = sum(1 for line in lines if re.match(r"^\s*\d+\.\s+", line))
    code_block_count = text.count("```")
    has_json = 0.0
    try:
        json.loads(text)
        has_json = 1.0
    except Exception:
        has_json = 0.0

    avg_line_len = 0.0
    if lines:
        avg_line_len = sum(len(line) for line in lines) / line_count

    return {
        "line_count": float(line_count),
        "bullet_ratio": bullet_count / line_count if line_count else 0.0,
        "numbered_ratio": numbered_count / line_count if line_count else 0.0,
        "code_block_count": float(code_block_count),
        "has_json": has_json,
        "avg_line_len": avg_line_len,
    }


def structural_similarity(a: str, b: str) -> float:
    sig_a = structural_signature(a)
    sig_b = structural_signature(b)
    scores = []
    for key in ("line_count", "bullet_ratio", "numbered_ratio", "avg_line_len"):
        value_a = sig_a[key]
        value_b = sig_b[key]
        denom = max(value_a, value_b, 1.0)
        scores.append(1.0 - min(abs(value_a - value_b) / denom, 1.0))

    for key in ("code_block_count", "has_json"):
        scores.append(1.0 if sig_a[key] == sig_b[key] else 0.0)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def combined_similarity(a: str, b: str) -> Dict[str, float]:
    seq = sequence_similarity(a, b)
    jac = jaccard_similarity(a, b)
    struct = structural_similarity(a, b)
    combined = (0.5 * seq) + (0.3 * jac) + (0.2 * struct)
    return {
        "sequence": seq,
        "jaccard": jac,
        "structure": struct,
        "combined": combined,
    }


def average_pairwise_similarity(texts: list[str]) -> Dict[str, float]:
    if len(texts) < 2:
        return {"combined": 1.0, "sequence": 1.0, "jaccard": 1.0, "structure": 1.0}
    total = {"combined": 0.0, "sequence": 0.0, "jaccard": 0.0, "structure": 0.0}
    count = 0
    for idx in range(len(texts)):
        for jdx in range(idx + 1, len(texts)):
            sim = combined_similarity(texts[idx], texts[jdx])
            for key in total:
                total[key] += sim[key]
            count += 1
    return {key: value / count for key, value in total.items()}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
