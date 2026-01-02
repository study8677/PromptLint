from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from promptlint.core.types import AggregateScore


class ReportRenderer(ABC):
    """Render aggregate scores into a report format."""

    name: str

    @abstractmethod
    def render(self, scores: Sequence[AggregateScore]) -> str:
        """Return a report string."""
