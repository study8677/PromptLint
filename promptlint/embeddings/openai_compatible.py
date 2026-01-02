from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from promptlint.core.types import EmbeddingConfig, ProviderConfig


class EmbeddingError(Exception):
    """Non-retryable embedding error."""


class RetryableEmbeddingError(EmbeddingError):
    """Retryable embedding error."""


class OpenAICompatibleEmbedder:
    """Embedding client for OpenAI-compatible providers."""

    def __init__(self, provider: ProviderConfig, config: EmbeddingConfig) -> None:
        self.provider = provider
        self.config = config
        self._client = httpx.AsyncClient()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        api_base = _resolve_api_base(self.provider)
        api_key = _resolve_api_key(self.provider)
        endpoint = self.provider.metadata.get("embedding_endpoint", "embeddings")
        url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.provider.metadata.get("headers", {}) or {})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "input": texts,
        }

        max_retries = self.provider.max_retries
        timeout = self.provider.timeout_s or 60
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(min=1, max=20),
            retry=retry_if_exception_type(RetryableEmbeddingError),
            reraise=True,
        )

        async for attempt in retrying:
            with attempt:
                try:
                    response = await self._client.post(
                        url, headers=headers, json=payload, timeout=timeout
                    )
                except httpx.RequestError as exc:
                    raise RetryableEmbeddingError(str(exc)) from exc

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RetryableEmbeddingError(
                        f"HTTP {response.status_code}: {response.text}"
                    )
                if response.status_code >= 400:
                    raise EmbeddingError(
                        f"HTTP {response.status_code}: {response.text}"
                    )

                data = response.json()
                return _extract_embeddings(data)

        raise EmbeddingError("Embedding request failed")

    async def aclose(self) -> None:
        await self._client.aclose()


def _resolve_api_base(config: ProviderConfig) -> str:
    if config.api_base:
        return config.api_base
    if config.name == "openai":
        return "https://api.openai.com/v1"
    raise RuntimeError(f"api_base is required for provider {config.name}")


def _resolve_api_key(config: ProviderConfig) -> Optional[str]:
    if not config.api_key_env:
        return None
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key env var: {config.api_key_env} for provider {config.name}"
        )
    return api_key


def _extract_embeddings(payload: Dict[str, Any]) -> List[List[float]]:
    data = payload.get("data") or []
    return [item.get("embedding", []) for item in data]
