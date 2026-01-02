import unittest

from promptlint.utils.similarity import average_pairwise_similarity


class SimilarityTests(unittest.TestCase):
    def test_average_pairwise_similarity_single(self) -> None:
        result = average_pairwise_similarity(["hello"])
        self.assertAlmostEqual(result["combined"], 1.0)

    def test_average_pairwise_similarity_pair(self) -> None:
        result = average_pairwise_similarity(["hello", "hello"])
        self.assertAlmostEqual(result["combined"], 1.0)


if __name__ == "__main__":
    unittest.main()
