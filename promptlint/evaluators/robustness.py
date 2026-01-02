from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from promptlint.core.types import Measure, Prompt, RunResult
from promptlint.evaluators.base import Evaluator
from promptlint.evaluators.constraints import evaluate_constraint
from promptlint.utils.embeddings import average_pairwise_cosine
from promptlint.utils.similarity import average_pairwise_similarity, clamp


class RobustnessEvaluator(Evaluator):
    """Default evaluator producing robustness-related measures."""

    name = "robustness"

    def __init__(self) -> None:
        self.weights = {
            "constraint_adherence": 0.3,
            "cross_model_consistency": 0.3,
            "cross_temperature_stability": 0.2,
            "task_alignment": 0.1,
            "success_rate": 0.1,
        }

    def evaluate(self, prompt: Prompt, runs: Sequence[RunResult]) -> Sequence[Measure]:
        measures: List[Measure] = []

        constraint_value, constraint_details = self._constraint_adherence(prompt, runs)
        constraint_weight = (
            self.weights["constraint_adherence"]
            if constraint_details["constraint_count"] > 0
            else 0.0
        )
        measures.append(
            Measure(
                name="constraint_adherence",
                value=constraint_value,
                details={**constraint_details, "weight": constraint_weight},
            )
        )

        cm_value, cm_details = self._cross_model_consistency(runs)
        measures.append(
            Measure(
                name="cross_model_consistency",
                value=cm_value,
                details={**cm_details, "weight": self.weights["cross_model_consistency"]},
            )
        )

        st_value, st_details = self._cross_temperature_stability(runs)
        measures.append(
            Measure(
                name="cross_temperature_stability",
                value=st_value,
                details={**st_details, "weight": self.weights["cross_temperature_stability"]},
            )
        )

        ta_value, ta_details = self._task_alignment(prompt, runs, constraint_value)
        ta_weight = (
            self.weights["task_alignment"] if ta_details["has_signal"] else 0.0
        )
        measures.append(
            Measure(
                name="task_alignment",
                value=ta_value,
                details={**ta_details, "weight": ta_weight},
            )
        )

        sr_value, sr_details = self._success_rate(runs)
        measures.append(
            Measure(
                name="success_rate",
                value=sr_value,
                details={**sr_details, "weight": self.weights["success_rate"]},
            )
        )

        return measures

    def _constraint_adherence(
        self, prompt: Prompt, runs: Sequence[RunResult]
    ) -> Tuple[float, Dict[str, object]]:
        if not runs:
            return 0.0, {"constraint_count": len(prompt.constraints)}

        if not prompt.constraints:
            return 1.0, {"constraint_count": 0, "note": "no constraints"}

        per_run_scores = []
        per_constraint = {constraint.name: [] for constraint in prompt.constraints}
        for run in runs:
            weights = []
            scores = []
            for constraint in prompt.constraints:
                result = evaluate_constraint(run.output.text, constraint)
                per_constraint[constraint.name].append(result.score)
                weights.append(constraint.weight)
                scores.append(result.score)

            weighted = _weighted_average(scores, weights)
            per_run_scores.append(weighted)

        overall = sum(per_run_scores) / len(per_run_scores)
        details: Dict[str, object] = {
            "constraint_count": len(prompt.constraints),
            "per_run_avg": overall,
        }
        for name, values in per_constraint.items():
            if values:
                details[f"constraint_{name}"] = sum(values) / len(values)
        return clamp(overall), details

    def _cross_model_consistency(
        self, runs: Sequence[RunResult]
    ) -> Tuple[float, Dict[str, object]]:
        by_sampling: Dict[Tuple[float, float, int], List[RunResult]] = defaultdict(list)
        for run in runs:
            key = (
                run.sampling.temperature,
                run.sampling.top_p,
                run.sampling.max_tokens,
            )
            by_sampling[key].append(run)

        if not by_sampling:
            return 0.0, {"groups": 0}

        scores = []
        text_scores = []
        embedding_scores = []
        for runs_group in by_sampling.values():
            score, details = _group_similarity(runs_group)
            scores.append(score)
            if details.get("text") is not None:
                text_scores.append(details["text"])
            if details.get("semantic") is not None:
                embedding_scores.append(details["semantic"])

        overall = sum(scores) / len(scores)
        details: Dict[str, object] = {"groups": len(by_sampling)}
        if text_scores:
            details["text_similarity"] = sum(text_scores) / len(text_scores)
        if embedding_scores:
            details["semantic_similarity"] = sum(embedding_scores) / len(embedding_scores)
        return clamp(overall), details

    def _cross_temperature_stability(
        self, runs: Sequence[RunResult]
    ) -> Tuple[float, Dict[str, object]]:
        by_model: Dict[str, List[RunResult]] = defaultdict(list)
        for run in runs:
            model_key = f"{run.model.provider}:{run.model.name}"
            by_model[model_key].append(run)

        if not by_model:
            return 0.0, {"models": 0}

        scores = []
        text_scores = []
        embedding_scores = []
        for runs_group in by_model.values():
            score, details = _group_similarity(runs_group)
            scores.append(score)
            if details.get("text") is not None:
                text_scores.append(details["text"])
            if details.get("semantic") is not None:
                embedding_scores.append(details["semantic"])

        overall = sum(scores) / len(scores)
        details: Dict[str, object] = {"models": len(by_model)}
        if text_scores:
            details["text_similarity"] = sum(text_scores) / len(text_scores)
        if embedding_scores:
            details["semantic_similarity"] = sum(embedding_scores) / len(embedding_scores)
        return clamp(overall), details

    def _task_alignment(
        self, prompt: Prompt, runs: Sequence[RunResult], fallback: float
    ) -> Tuple[float, Dict[str, object]]:
        expected_format = (prompt.metadata.get("expected_format") or "").lower()
        if not expected_format:
            has_signal = bool(prompt.constraints)
            return (
                clamp(fallback),
                {"has_signal": has_signal, "note": "fallback_to_constraints"},
            )

        scores = []
        for run in runs:
            scores.append(1.0 if _matches_format(run.output.text, expected_format) else 0.0)

        if not scores:
            return 0.0, {"has_signal": True, "expected_format": expected_format}

        return (
            sum(scores) / len(scores),
            {"has_signal": True, "expected_format": expected_format},
        )

    def _success_rate(self, runs: Sequence[RunResult]) -> Tuple[float, Dict[str, object]]:
        if not runs:
            return 0.0, {"total": 0, "failure_rate": 1.0}
        failures = sum(1 for run in runs if not run.success)
        total = len(runs)
        failure_rate = failures / total if total else 1.0
        return (1.0 - failure_rate), {"total": total, "failure_rate": failure_rate}


def _weighted_average(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weights) if weights else 0.0
    if total_weight == 0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def _group_similarity(runs: Sequence[RunResult]) -> Tuple[float, Dict[str, float]]:
    outputs = [run.output.text for run in runs]
    text_sim = average_pairwise_similarity(outputs)["combined"]
    vectors = [run.metadata.get("embedding") for run in runs if run.metadata.get("embedding")]
    if len(vectors) >= 2:
        emb_sim = average_pairwise_cosine(vectors)
        combined = clamp((0.7 * emb_sim) + (0.3 * text_sim))
        return combined, {"text": text_sim, "semantic": emb_sim}
    return clamp(text_sim), {"text": text_sim}


def _matches_format(text: str, expected_format: str) -> bool:
    if expected_format in {"json", "json_object"}:
        return _is_json(text)
    if expected_format in {"bullets", "bullet", "list"}:
        return _has_bullets(text)
    if expected_format in {"numbered_list", "numbered", "steps"}:
        return _has_numbered_list(text)
    if expected_format == "code":
        return "```" in text
    if expected_format == "table":
        return "|" in text and "---" in text
    return True


def _is_json(text: str) -> bool:
    import json

    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _has_bullets(text: str) -> bool:
    return any(line.strip().startswith(('-', '*')) for line in text.splitlines())


def _has_numbered_list(text: str) -> bool:
    for line in text.splitlines():
        if line.strip().startswith(tuple(str(i) + "." for i in range(1, 10))):
            return True
    return False
