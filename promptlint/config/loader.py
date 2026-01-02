from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from promptlint.core.types import (
    Constraint,
    EmbeddingConfig,
    ModelSpec,
    Prompt,
    ProviderConfig,
    RunConfig,
    SamplingConfig,
)
from promptlint.ladder.ladder import ModelLadder


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    prompts: Sequence[Prompt]
    ladder: ModelLadder
    sampling: Sequence[SamplingConfig]
    providers: Dict[str, ProviderConfig]
    run: RunConfig


def load_suite(path: str) -> SuiteSpec:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load suite configs") from exc

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    suite_name = data.get("name", "unnamed-suite")
    providers_cfg = data.get("providers", [])
    providers: Dict[str, ProviderConfig] = {}
    for provider in providers_cfg:
        config = ProviderConfig(
            name=provider.get("name", ""),
            kind=provider.get("kind", "openai_compatible"),
            api_base=provider.get("api_base"),
            api_key_env=provider.get("api_key_env"),
            timeout_s=provider.get("timeout_s"),
            max_retries=int(provider.get("max_retries", 4)),
            price_per_1k_prompt=_parse_optional_float(provider.get("price_per_1k_prompt")),
            price_per_1k_completion=_parse_optional_float(
                provider.get("price_per_1k_completion")
            ),
            metadata=provider.get("metadata", {}) or {},
        )
        if config.name:
            providers[config.name] = config

    ladder_cfg = data.get("ladder", {})
    models_cfg = ladder_cfg.get("models", [])
    models = [
        ModelSpec(
            provider=model.get("provider", ""),
            name=model.get("name", ""),
            tier=int(model.get("tier", 0)),
            context_window=model.get("context_window"),
            metadata=model.get("metadata", {}) or {},
        )
        for model in models_cfg
    ]
    ladder = ModelLadder(name=ladder_cfg.get("name", "default"), models=models)

    prompt_cfgs = data.get("prompts", [])
    prompts: List[Prompt] = []
    for prompt in prompt_cfgs:
        constraints_cfg = prompt.get("constraints", [])
        constraints = [
            Constraint(
                name=constraint.get("name", ""),
                description=constraint.get("description", ""),
                kind=constraint.get("kind", "semantic"),
                weight=float(constraint.get("weight", 1.0)),
                rules=constraint.get("rules", {}) or {},
            )
            for constraint in constraints_cfg
        ]
        prompts.append(
            Prompt(
                id=prompt.get("id", ""),
                text=prompt.get("text", ""),
                constraints=constraints,
                metadata=prompt.get("metadata", {}) or {},
            )
        )

    sampling_cfgs = data.get("sampling", [])
    sampling: List[SamplingConfig] = []
    for item in sampling_cfgs:
        sampling.append(
            SamplingConfig(
                temperature=float(item.get("temperature", 0.2)),
                top_p=float(item.get("top_p", 1.0)),
                max_tokens=int(item.get("max_tokens", 512)),
                seed=item.get("seed"),
            )
        )

    if not sampling:
        sampling = [SamplingConfig()]

    run_cfg = data.get("run", {}) or {}
    embeddings_cfg = run_cfg.get("embeddings")
    embeddings = None
    if isinstance(embeddings_cfg, dict) and embeddings_cfg.get("provider") and embeddings_cfg.get("model"):
        embeddings = EmbeddingConfig(
            provider=embeddings_cfg.get("provider", ""),
            model=embeddings_cfg.get("model", ""),
            batch_size=int(embeddings_cfg.get("batch_size", 16)),
        )
    run = RunConfig(
        concurrency=int(run_cfg.get("concurrency", 8)),
        cache_path=run_cfg.get("cache_path", ".promptlint/cache.sqlite"),
        use_cache=bool(run_cfg.get("use_cache", True)),
        embeddings=embeddings,
    )

    return SuiteSpec(
        name=suite_name,
        prompts=prompts,
        ladder=ladder,
        sampling=sampling,
        providers=providers,
        run=run,
    )


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
