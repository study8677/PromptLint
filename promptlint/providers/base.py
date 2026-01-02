from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from promptlint.core.types import ModelOutput, ModelSpec, Prompt, ProviderConfig, SamplingConfig


class ModelProvider(ABC):
    """Provider interface for model execution."""

    name: str

    def __init__(self, config: Optional[ProviderConfig] = None) -> None:
        self.config = config

    @abstractmethod
    async def generate_async(
        self, prompt: Prompt, model: ModelSpec, sampling: SamplingConfig
    ) -> ModelOutput:
        """Run a single prompt through a model and return raw output."""

    def generate(
        self, prompt: Prompt, model: ModelSpec, sampling: SamplingConfig
    ) -> ModelOutput:
        """Synchronous wrapper for generate_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("generate() called inside a running event loop")
        return asyncio.run(self.generate_async(prompt, model, sampling))

    def supports(self, model: ModelSpec) -> bool:
        return model.provider == self.name

    def list_models(self) -> Sequence[ModelSpec]:
        return []

    def with_config(self, config: ProviderConfig) -> "ModelProvider":
        self.config = config
        return self

    async def aclose(self) -> None:
        """Close any underlying resources."""
        return None
