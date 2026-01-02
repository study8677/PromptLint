from __future__ import annotations

from typing import Optional

from promptlint.core.types import EmbeddingConfig, ProviderConfig
from promptlint.embeddings.openai_compatible import OpenAICompatibleEmbedder


def build_embedder(
    providers: dict[str, ProviderConfig], config: Optional[EmbeddingConfig]
) -> Optional[OpenAICompatibleEmbedder]:
    if not config:
        return None
    provider = providers.get(config.provider)
    if not provider:
        raise RuntimeError(f"Embedding provider not configured: {config.provider}")
    if provider.kind != "openai_compatible":
        raise RuntimeError(f"Unsupported embedding provider kind: {provider.kind}")
    return OpenAICompatibleEmbedder(provider, config)
