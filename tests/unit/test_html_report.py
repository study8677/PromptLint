import unittest

from promptlint.core.types import AggregateScore
from promptlint.reporting.html_report import HtmlReport


class HtmlReportTests(unittest.TestCase):
    def test_html_report_contains_prompt(self) -> None:
        report = HtmlReport()
        scores = [
            AggregateScore(
                prompt_id="p1",
                overall=0.9,
                components={"a": 0.9},
                details={"method": "weighted_geometric_mean"},
            )
        ]
        html = report.render(scores)
        self.assertIn("PromptLint Report", html)
        self.assertIn("p1", html)
        self.assertIn("<html", html)


if __name__ == "__main__":
    unittest.main()
