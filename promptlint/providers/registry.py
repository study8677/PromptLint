from __future__ import annotations

from typing import Dict, Mapping

from promptlint.core.types import ProviderConfig
from promptlint.providers.base import ModelProvider
from promptlint.providers.openai_compatible import OpenAICompatibleProvider


def build_providers(configs: Mapping[str, ProviderConfig]) -> Dict[str, ModelProvider]:
    providers: Dict[str, ModelProvider] = {}
    for name, config in configs.items():
        if config.kind == "openai_compatible":
            providers[name] = OpenAICompatibleProvider(config)
        else:
            raise RuntimeError(f"Unknown provider kind: {config.kind}")
    return providers


async def close_providers(providers: Mapping[str, ModelProvider]) -> None:
    for provider in providers.values():
        await provider.aclose()
