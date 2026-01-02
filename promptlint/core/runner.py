from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from promptlint.aggregators.base import Aggregator
from promptlint.cache.sqlite_cache import CacheStore
from promptlint.config.loader import SuiteSpec
from promptlint.core.types import AggregateScore, ModelOutput, Measure, Prompt, RunResult
from promptlint.embeddings.openai_compatible import OpenAICompatibleEmbedder
from promptlint.evaluators.base import Evaluator
from promptlint.providers.base import ModelProvider


@dataclass(frozen=True)
class PromptResult:
    prompt: Prompt
    runs: Sequence[RunResult]
    measures: Sequence[Measure]
    score: AggregateScore


@dataclass(frozen=True)
class SuiteResult:
    suite: SuiteSpec
    prompt_results: Sequence[PromptResult]


class SuiteRunner:
    def __init__(
        self,
        providers: Dict[str, ModelProvider],
        evaluator: Evaluator,
        aggregator: Aggregator,
        cache: Optional[CacheStore] = None,
        embedder: Optional[OpenAICompatibleEmbedder] = None,
        concurrency: int = 8,
        show_progress: bool = True,
    ) -> None:
        self.providers = providers
        self.evaluator = evaluator
        self.aggregator = aggregator
        self.cache = cache
        self.embedder = embedder
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.show_progress = show_progress

    async def run(self, suite: SuiteSpec) -> SuiteResult:
        tasks = []
        for prompt in suite.prompts:
            for model in suite.ladder.ordered():
                provider = self.providers.get(model.provider)
                if not provider:
                    raise RuntimeError(f"Provider not configured: {model.provider}")
                for sampling in suite.sampling:
                    tasks.append(
                        self._run_single(
                            suite=suite,
                            prompt=prompt,
                            provider=provider,
                            model=model,
                            sampling=sampling,
                        )
                    )

        total = len(tasks)
        runs_by_prompt: Dict[str, List[RunResult]] = {prompt.id: [] for prompt in suite.prompts}
        progress = tqdm(total=total, desc="Runs", disable=not self.show_progress)
        for task in asyncio.as_completed(tasks):
            run = await task
            runs_by_prompt[run.prompt_id].append(run)
            progress.update(1)
        progress.close()

        if self.embedder:
            runs_by_prompt = await self._attach_embeddings(runs_by_prompt)

        prompt_results: List[PromptResult] = []
        for prompt in suite.prompts:
            runs = runs_by_prompt.get(prompt.id, [])
            measures = self.evaluator.evaluate(prompt, runs)
            score = self.aggregator.aggregate(prompt, measures)
            cost_values = [run.cost_usd for run in runs if run.cost_usd is not None]
            cost_total = sum(cost_values) if cost_values else None
            if cost_total is not None:
                score = replace(
                    score, details={**score.details, "cost_usd": cost_total}
                )
            prompt_results.append(
                PromptResult(prompt=prompt, runs=runs, measures=measures, score=score)
            )

        return SuiteResult(suite=suite, prompt_results=prompt_results)

    async def _run_single(
        self,
        suite: SuiteSpec,
        prompt: Prompt,
        provider: ModelProvider,
        model,
        sampling,
    ) -> RunResult:
        cache_key = _run_cache_key(prompt, model, sampling, provider.config)
        if self.cache and suite.run.use_cache:
            cached = await self.cache.get_run(cache_key)
            if cached:
                return _run_from_cache(prompt.id, model, sampling, cached)

        async with self.semaphore:
            start = time.perf_counter()
            try:
                output = await provider.generate_async(prompt, model, sampling)
                duration_ms = int((time.perf_counter() - start) * 1000)
                usage = output.usage
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
                cost_usd = _estimate_cost(prompt_tokens, completion_tokens, model, provider)

                run = RunResult(
                    prompt_id=prompt.id,
                    model=model,
                    sampling=sampling,
                    output=output,
                    duration_ms=duration_ms,
                    success=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    metadata={"cached": False},
                )
            except Exception as exc:
                duration_ms = int((time.perf_counter() - start) * 1000)
                run = RunResult(
                    prompt_id=prompt.id,
                    model=model,
                    sampling=sampling,
                    output=ModelOutput(text=""),
                    duration_ms=duration_ms,
                    success=False,
                    metadata={"cached": False, "error": str(exc)},
                )

        if self.cache and suite.run.use_cache and run.success:
            await self.cache.set_run(cache_key, _run_to_cache(run))

        return run

    async def _attach_embeddings(
        self, runs_by_prompt: Dict[str, List[RunResult]]
    ) -> Dict[str, List[RunResult]]:
        if not self.embedder:
            return runs_by_prompt

        text_index: Dict[str, str] = {}
        for runs in runs_by_prompt.values():
            for run in runs:
                if run.output.text:
                    text_index[_text_hash(run.output.text)] = run.output.text

        embeddings: Dict[str, List[float]] = {}
        missing_texts: List[str] = []
        for text_hash, text in text_index.items():
            cache_key = _embedding_cache_key(self.embedder.config.model, text_hash)
            if self.cache:
                cached = await self.cache.get_embedding(cache_key)
            else:
                cached = None
            if cached and "vector" in cached:
                embeddings[text_hash] = cached["vector"]
            else:
                missing_texts.append(text)

        if missing_texts:
            batch_size = self.embedder.config.batch_size
            for idx in range(0, len(missing_texts), batch_size):
                batch = missing_texts[idx : idx + batch_size]
                vectors = await self.embedder.embed(batch)
                for text, vector in zip(batch, vectors):
                    text_hash = _text_hash(text)
                    embeddings[text_hash] = vector
                    if self.cache:
                        cache_key = _embedding_cache_key(
                            self.embedder.config.model, text_hash
                        )
                        await self.cache.set_embedding(
                            cache_key, {"model": self.embedder.config.model, "vector": vector}
                        )

        updated: Dict[str, List[RunResult]] = {}
        for prompt_id, runs in runs_by_prompt.items():
            new_runs: List[RunResult] = []
            for run in runs:
                metadata = dict(run.metadata)
                if run.output.text:
                    text_hash = _text_hash(run.output.text)
                    vector = embeddings.get(text_hash)
                    if vector is not None:
                        metadata["embedding"] = vector
                        metadata["embedding_model"] = self.embedder.config.model
                new_runs.append(replace(run, metadata=metadata))
            updated[prompt_id] = new_runs
        return updated


def _run_cache_key(prompt: Prompt, model, sampling, provider_config) -> str:
    provider_payload = {}
    if provider_config is not None:
        provider_payload = {
            "name": provider_config.name,
            "api_base": provider_config.api_base,
            "endpoint": provider_config.metadata.get("endpoint"),
            "params": provider_config.metadata.get("params"),
        }
    payload = {
        "prompt_id": prompt.id,
        "prompt_text": prompt.text,
        "provider": model.provider,
        "model": model.name,
        "sampling": {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
        },
        "provider_config": provider_payload,
    }
    return _hash_payload(payload)


def _embedding_cache_key(model: str, text_hash: str) -> str:
    return _hash_payload({"model": model, "text_hash": text_hash})


def _hash_payload(payload: Dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_to_cache(run: RunResult) -> Dict[str, object]:
    return {
        "output_text": run.output.text,
        "raw": run.output.raw,
        "usage": run.output.usage,
        "duration_ms": run.duration_ms,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "total_tokens": run.total_tokens,
        "cost_usd": run.cost_usd,
    }


def _run_from_cache(prompt_id: str, model, sampling, cached: Dict[str, object]) -> RunResult:
    output = ModelOutput(
        text=str(cached.get("output_text", "")),
        raw=cached.get("raw"),
        usage=cached.get("usage") or {},
    )
    return RunResult(
        prompt_id=prompt_id,
        model=model,
        sampling=sampling,
        output=output,
        duration_ms=_to_int(cached.get("duration_ms")),
        success=True,
        prompt_tokens=_to_int(cached.get("prompt_tokens")),
        completion_tokens=_to_int(cached.get("completion_tokens")),
        total_tokens=_to_int(cached.get("total_tokens")),
        cost_usd=_to_float(cached.get("cost_usd")),
        metadata={"cached": True},
    )


def _estimate_cost(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    model,
    provider: ModelProvider,
) -> Optional[float]:
    if prompt_tokens is None and completion_tokens is None:
        return None
    config = provider.config
    if not config:
        return None

    price_prompt = _pricing_value(model, config, "price_per_1k_prompt")
    price_completion = _pricing_value(model, config, "price_per_1k_completion")
    if price_prompt is None and price_completion is None:
        return None

    cost = 0.0
    if price_prompt is not None and prompt_tokens is not None:
        cost += price_prompt * (prompt_tokens / 1000.0)
    if price_completion is not None and completion_tokens is not None:
        cost += price_completion * (completion_tokens / 1000.0)
    return cost


def _pricing_value(model, config, key: str) -> Optional[float]:
    if isinstance(model.metadata, dict) and key in model.metadata:
        try:
            return float(model.metadata[key])
        except (TypeError, ValueError):
            return None
    value = getattr(config, key)
    return float(value) if value is not None else None


def _to_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
