import unittest

from promptlint.core.types import Constraint
from promptlint.evaluators.constraints import evaluate_constraint


class ConstraintTests(unittest.TestCase):
    def test_count_rule(self) -> None:
        constraint = Constraint(
            name="bullets",
            description="",
            kind="format",
            weight=1.0,
            rules={"type": "count", "pattern": r"^\s*[-*]\s+", "exact": 2},
        )
        text = "- one\n- two\n"
        result = evaluate_constraint(text, constraint)
        self.assertEqual(result.score, 1.0)

    def test_json_rule(self) -> None:
        constraint = Constraint(
            name="json",
            description="",
            kind="format",
            weight=1.0,
            rules={"type": "json", "expect": "object"},
        )
        result = evaluate_constraint('{"a": 1}', constraint)
        self.assertEqual(result.score, 1.0)

    def test_all_lines_match_rule(self) -> None:
        constraint = Constraint(
            name="lines",
            description="",
            kind="structure",
            weight=1.0,
            rules={"type": "all_lines_match", "pattern": r"^\s*\d+\.",},
        )
        text = "1. a\n2. b\n"
        result = evaluate_constraint(text, constraint)
        self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
