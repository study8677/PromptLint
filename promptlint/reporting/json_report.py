from __future__ import annotations

import json
from typing import Sequence

from promptlint.core.types import AggregateScore
from promptlint.reporting.base import ReportRenderer


class JsonReport(ReportRenderer):
    name = "json"

    def render(self, scores: Sequence[AggregateScore]) -> str:
        payload = [
            {
                "prompt_id": score.prompt_id,
                "overall": score.overall,
                "components": score.components,
                "details": score.details,
            }
            for score in scores
        ]
        return json.dumps(payload, ensure_ascii=False, indent=2)
