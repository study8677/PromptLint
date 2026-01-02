from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from promptlint.core.types import ModelSpec


@dataclass(frozen=True)
class ModelLadder:
    """Ordered ladder of models from large to small (higher tier -> larger)."""

    name: str
    models: Sequence[ModelSpec]

    def ordered(self) -> Sequence[ModelSpec]:
        return sorted(self.models, key=lambda model: model.tier, reverse=True)
