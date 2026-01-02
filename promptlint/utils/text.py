from __future__ import annotations

import re
from typing import Iterable, List


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(text))


def non_empty_lines(text: str) -> List[str]:
    return [line for line in text.splitlines() if line.strip()]


def count_lines_matching(lines: Iterable[str], pattern: str) -> int:
    regex = re.compile(pattern)
    return sum(1 for line in lines if regex.search(line))
