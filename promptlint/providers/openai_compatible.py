from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from promptlint.core.types import ModelOutput, ModelSpec, Prompt, ProviderConfig, SamplingConfig
from promptlint.providers.base import ModelProvider


class ProviderError(Exception):
    """Non-retryable provider error."""


class RetryableProviderError(ProviderError):
    """Retryable provider error."""


class OpenAICompatibleProvider(ModelProvider):
    """OpenAI-compatible provider using /chat/completions."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.name = config.name
        self._client = httpx.AsyncClient()

    async def generate_async(
        self, prompt: Prompt, model: ModelSpec, sampling: SamplingConfig
    ) -> ModelOutput:
        config = self._require_config()
        api_base = _resolve_api_base(config)
        api_key = _resolve_api_key(config)
        endpoint = config.metadata.get("endpoint", "chat/completions")
        url = f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(config.metadata.get("headers", {}) or {})

        payload: Dict[str, Any] = {
            "model": model.name,
            "messages": [{"role": "user", "content": prompt.text}],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.seed is not None:
            payload["seed"] = sampling.seed
        payload.update(config.metadata.get("params", {}) or {})

        response_payload = await self._post_json(
            url=url,
            headers=headers,
            payload=payload,
            timeout=config.timeout_s or 60,
            max_retries=config.max_retries,
        )

        output_text = _extract_text(response_payload)
        usage = _extract_usage(response_payload)
        return ModelOutput(text=output_text, raw={"response": response_payload}, usage=usage)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _require_config(self) -> ProviderConfig:
        if self.config is None:
            raise RuntimeError("Provider config is required")
        return self.config

    async def _post_json(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
        max_retries: int,
    ) -> Dict[str, Any]:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(min=1, max=20),
            retry=retry_if_exception_type(RetryableProviderError),
            reraise=True,
        )

        async for attempt in retrying:
            with attempt:
                try:
                    response = await self._client.post(
                        url, headers=headers, json=payload, timeout=timeout
                    )
                except httpx.RequestError as exc:
                    raise RetryableProviderError(str(exc)) from exc

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RetryableProviderError(
                        f"HTTP {response.status_code}: {response.text}"
                    )
                if response.status_code >= 400:
                    raise ProviderError(
                        f"HTTP {response.status_code}: {response.text}"
                    )

                return response.json()

        raise ProviderError("Provider request failed")


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


def _extract_text(payload: Dict[str, Any]) -> str:
    if "choices" in payload and payload["choices"]:
        choice = payload["choices"][0]
        message = choice.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message["content"])
        if choice.get("text") is not None:
            return str(choice["text"])
    if payload.get("output_text") is not None:
        return str(payload["output_text"])
    if payload.get("content") is not None:
        return str(payload["content"])
    return ""


def _extract_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    usage = payload.get("usage") or {}
    result: Dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            result[key] = value
    return result
