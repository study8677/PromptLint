from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from promptlint.core.types import AggregateScore, Measure, Prompt, RunResult
from promptlint.core.runner import PromptResult, SuiteResult


def run_result_to_dict(run: RunResult, include_raw: bool = False) -> Dict[str, Any]:
    output = {"text": run.output.text}
    if include_raw:
        output["raw"] = _ensure_jsonable(run.output.raw)
    if run.output.usage:
        output["usage"] = run.output.usage

    return {
        "prompt_id": run.prompt_id,
        "model": asdict(run.model),
        "sampling": asdict(run.sampling),
        "output": output,
        "duration_ms": run.duration_ms,
        "success": run.success,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "total_tokens": run.total_tokens,
        "cost_usd": run.cost_usd,
        "metadata": run.metadata,
    }


def measure_to_dict(measure: Measure) -> Dict[str, Any]:
    return {
        "name": measure.name,
        "value": measure.value,
        "details": measure.details,
    }


def aggregate_to_dict(score: AggregateScore) -> Dict[str, Any]:
    return {
        "prompt_id": score.prompt_id,
        "overall": score.overall,
        "components": score.components,
        "details": score.details,
    }


def prompt_to_dict(prompt: Prompt) -> Dict[str, Any]:
    return {
        "id": prompt.id,
        "text": prompt.text,
        "constraints": [asdict(constraint) for constraint in prompt.constraints],
        "metadata": prompt.metadata,
    }


def prompt_result_to_dict(
    result: PromptResult, include_raw: bool = False
) -> Dict[str, Any]:
    return {
        "prompt": prompt_to_dict(result.prompt),
        "runs": [run_result_to_dict(run, include_raw=include_raw) for run in result.runs],
        "measures": [measure_to_dict(measure) for measure in result.measures],
        "score": aggregate_to_dict(result.score),
    }


def suite_result_to_dict(
    result: SuiteResult, include_raw: bool = False
) -> Dict[str, Any]:
    return {
        "suite": {
            "name": result.suite.name,
            "ladder": result.suite.ladder.name,
            "models": [asdict(model) for model in result.suite.ladder.ordered()],
            "sampling": [asdict(sampling) for sampling in result.suite.sampling],
            "providers": [
                asdict(config) for config in result.suite.providers.values()
            ],
            "run": asdict(result.suite.run),
        },
        "prompts": [
            prompt_result_to_dict(prompt_result, include_raw=include_raw)
            for prompt_result in result.prompt_results
        ],
    }


def suite_result_to_json(result: SuiteResult, include_raw: bool = False) -> str:
    payload = suite_result_to_dict(result, include_raw=include_raw)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ensure_jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
