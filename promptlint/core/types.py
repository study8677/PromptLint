from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class Constraint:
    """A prompt-level constraint to evaluate adherence against."""

    name: str
    description: str
    kind: str = "semantic"
    weight: float = 1.0
    rules: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prompt:
    """Prompt spec with optional structured constraints."""

    id: str
    text: str
    constraints: Sequence[Constraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters for a model run."""

    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 512
    seed: Optional[int] = None


@dataclass(frozen=True)
class ProviderConfig:
    """Provider API configuration loaded from suite files."""

    name: str
    kind: str = "openai_compatible"
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_s: Optional[float] = None
    max_retries: int = 4
    price_per_1k_prompt: Optional[float] = None
    price_per_1k_completion: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration for semantic similarity."""

    provider: str
    model: str
    batch_size: int = 16


@dataclass(frozen=True)
class RunConfig:
    """Runtime execution settings."""

    concurrency: int = 8
    cache_path: Optional[str] = ".promptlint/cache.sqlite"
    use_cache: bool = True
    embeddings: Optional[EmbeddingConfig] = None


@dataclass(frozen=True)
class ModelSpec:
    """Model identity and ladder placement."""

    provider: str
    name: str
    tier: int
    context_window: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput:
    """Raw model output payload."""

    text: str
    raw: Optional[Any] = None
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """Single model run for a prompt at a sampling config."""

    prompt_id: str
    model: ModelSpec
    sampling: SamplingConfig
    output: ModelOutput
    duration_ms: Optional[int] = None
    success: bool = True
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Measure:
    """Metric value emitted by an evaluator."""

    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Set of measures for a prompt and run group."""

    prompt_id: str
    model: ModelSpec
    sampling: SamplingConfig
    measures: Sequence[Measure]


@dataclass(frozen=True)
class AggregateScore:
    """Aggregated score across measures for a prompt."""

    prompt_id: str
    overall: float
    components: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
