from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from promptlint.core.types import Measure, Prompt, RunResult


class Evaluator(ABC):
    """Evaluator interface producing measures from model runs."""

    name: str

    @abstractmethod
    def evaluate(self, prompt: Prompt, runs: Sequence[RunResult]) -> Sequence[Measure]:
        """Return measures for the prompt given a set of model runs."""
