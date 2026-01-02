import unittest

from promptlint.aggregators.robustness import RobustnessAggregator
from promptlint.core.types import Measure, Prompt


class AggregatorTests(unittest.TestCase):
    def test_geometric_mean_penalizes_low_score(self) -> None:
        aggregator = RobustnessAggregator()
        prompt = Prompt(id="p1", text="hello")
        measures = [
            Measure(name="a", value=1.0, details={"weight": 0.5}),
            Measure(name="b", value=0.2, details={"weight": 0.5}),
        ]
        score = aggregator.aggregate(prompt, measures)
        self.assertLess(score.overall, 0.6)
        self.assertIn("base_score", score.details)

    def test_penalty_applied_on_constraint(self) -> None:
        aggregator = RobustnessAggregator()
        prompt = Prompt(id="p2", text="hello")
        measures = [
            Measure(name="constraint_adherence", value=0.5, details={"weight": 1.0}),
        ]
        score = aggregator.aggregate(prompt, measures)
        self.assertLess(score.overall, 0.5)


if __name__ == "__main__":
    unittest.main()
