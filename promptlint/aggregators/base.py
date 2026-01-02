from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from promptlint.core.types import AggregateScore, Measure, Prompt


class Aggregator(ABC):
    """Combine measures into a prompt-level score."""

    name: str

    @abstractmethod
    def aggregate(self, prompt: Prompt, measures: Sequence[Measure]) -> AggregateScore:
        """Return an aggregate score and components."""
