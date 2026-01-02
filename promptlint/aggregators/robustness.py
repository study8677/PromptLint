from __future__ import annotations

import math
from typing import Sequence

from promptlint.aggregators.base import Aggregator
from promptlint.core.types import AggregateScore, Measure, Prompt
from promptlint.utils.similarity import clamp


class RobustnessAggregator(Aggregator):
    """Weighted geometric mean with stability penalties."""

    name = "robustness"

    def aggregate(self, prompt: Prompt, measures: Sequence[Measure]) -> AggregateScore:
        if not measures:
            return AggregateScore(prompt_id=prompt.id, overall=0.0, components={})

        total_weight = 0.0
        weighted_log_sum = 0.0
        components = {}
        weights = {}
        values = []
        for measure in measures:
            weight = float(measure.details.get("weight", 1.0))
            value = clamp(float(measure.value))
            if weight <= 0.0:
                continue
            total_weight += weight
            weighted_log_sum += weight * math.log(max(value, 1e-6))
            components[measure.name] = value
            weights[measure.name] = weight
            values.append(value)

        if not total_weight:
            return AggregateScore(prompt_id=prompt.id, overall=0.0, components=components)

        base_score = math.exp(weighted_log_sum / total_weight)
        penalty = _stability_penalty(components)
        overall = clamp(base_score * penalty)
        return AggregateScore(
            prompt_id=prompt.id,
            overall=overall,
            components=components,
            details={
                "method": "weighted_geometric_mean",
                "base_score": base_score,
                "penalty": penalty,
                "min_component": min(values) if values else None,
                "spread": (max(values) - min(values)) if len(values) > 1 else 0.0,
                "weights": weights,
            },
        )


def _stability_penalty(components: dict[str, float]) -> float:
    if not components:
        return 1.0

    values = list(components.values())
    spread = max(values) - min(values) if len(values) > 1 else 0.0
    penalty = 1.0 - min(0.3, spread * 0.4)

    constraint_value = components.get("constraint_adherence")
    if constraint_value is not None and constraint_value < 0.7:
        penalty *= 0.85

    return clamp(penalty)
