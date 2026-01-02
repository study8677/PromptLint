from __future__ import annotations

from typing import Sequence

from promptlint.core.types import AggregateScore
from promptlint.reporting.base import ReportRenderer


class HtmlReport(ReportRenderer):
    name = "html"

    def render(self, scores: Sequence[AggregateScore]) -> str:
        rows = []
        for score in scores:
            cost = score.details.get("cost_usd") if score.details else None
            rows.append(
                "<tr>"
                f"<td>{_escape(score.prompt_id)}</td>"
                f"<td>{score.overall:.3f}</td>"
                f"<td>{'' if cost is None else f'{cost:.4f}'}</td>"
                "</tr>"
            )

        sections = []
        for score in scores:
            sections.append(_render_prompt(score))

        return _wrap_html("\n".join(rows), "\n".join(sections))


def _render_prompt(score: AggregateScore) -> str:
    components = []
    for name in sorted(score.components):
        value = score.components[name]
        components.append(
            "<div class='component'>"
            f"<div class='label'>{_escape(name)}</div>"
            f"<div class='bar'><span style='width:{value * 100:.1f}%'></span></div>"
            f"<div class='value'>{value:.3f}</div>"
            "</div>"
        )

    details_items = []
    if score.details:
        for key, value in score.details.items():
            details_items.append(
                f"<div class='detail'><span>{_escape(str(key))}</span>: {_escape(str(value))}</div>"
            )

    return (
        "<section class='prompt'>"
        f"<h2>{_escape(score.prompt_id)}</h2>"
        f"<div class='overall'>Overall: {score.overall:.3f}</div>"
        "<div class='components'>"
        + "".join(components)
        + "</div>"
        + "<details>"
        + "<summary>Details</summary>"
        + "<div class='details'>"
        + "".join(details_items)
        + "</div>"
        + "</details>"
        + "</section>"
    )


def _wrap_html(table_rows: str, sections: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PromptLint Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f3ee;
      --card: #ffffff;
      --ink: #222222;
      --muted: #6b6b6b;
      --accent: #2d6cdf;
      --accent-2: #88b7ff;
    }}
    body {{
      font-family: "JetBrains Mono", "IBM Plex Mono", "Menlo", monospace;
      background: linear-gradient(135deg, #f6f3ee 0%, #f1efe9 100%);
      color: var(--ink);
      margin: 0;
      padding: 32px;
    }}
    header {{
      margin-bottom: 24px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }}
    th, td {{
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid #eee;
    }}
    th {{
      background: #f0f2f6;
      font-weight: 600;
    }}
    section.prompt {{
      margin-top: 24px;
      background: var(--card);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
    }}
    .overall {{
      font-size: 16px;
      margin-bottom: 12px;
    }}
    .components {{
      display: grid;
      gap: 12px;
    }}
    .component {{
      display: grid;
      grid-template-columns: 160px 1fr 60px;
      align-items: center;
      gap: 12px;
    }}
    .label {{
      color: var(--muted);
    }}
    .bar {{
      height: 10px;
      background: #e6e6e6;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    details {{
      margin-top: 16px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }}
    .details {{
      margin-top: 8px;
      display: grid;
      gap: 6px;
      color: var(--muted);
    }}
    @media (max-width: 720px) {{
      body {{
        padding: 20px;
      }}
      .component {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PromptLint Report</h1>
    <p>Robustness scores across models and temperatures.</p>
  </header>
  <table>
    <thead>
      <tr>
        <th>Prompt</th>
        <th>Overall</th>
        <th>Cost (USD)</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  {sections}
</body>
</html>"""


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&#39;")
    )
