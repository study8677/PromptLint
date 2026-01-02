from __future__ import annotations

from typing import Sequence

from promptlint.core.types import AggregateScore
from promptlint.reporting.base import ReportRenderer


class MarkdownReport(ReportRenderer):
    name = "markdown"

    def render(self, scores: Sequence[AggregateScore]) -> str:
        lines = ["# PromptLint Report", ""]
        for score in scores:
            lines.append(f"## Prompt: {score.prompt_id}")
            lines.append("")
            lines.append(f"- Overall: {score.overall:.3f}")
            for key in sorted(score.components):
                value = score.components[key]
                lines.append(f"- {key}: {value:.3f}")
            cost = score.details.get("cost_usd") if score.details else None
            if cost is not None:
                lines.append(f"- cost_usd: {cost:.4f}")
            lines.append("")
        return "\n".join(lines)
