import tempfile
import unittest

from promptlint.config.loader import load_suite


class ConfigLoaderTests(unittest.TestCase):
    def test_load_suite_with_run_and_provider(self) -> None:
        config = """
name: "test-suite"
providers:
  - name: "openai"
    kind: "openai_compatible"
    api_base: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    timeout_s: 30
    max_retries: 3
    price_per_1k_prompt: 0.001
    price_per_1k_completion: 0.002
run:
  concurrency: 4
  cache_path: ".promptlint/cache.sqlite"
  use_cache: true
  embeddings:
    provider: "openai"
    model: "text-embedding-3-small"
    batch_size: 8
ladder:
  name: "default"
  models:
    - provider: "openai"
      name: "gpt-5"
      tier: 5
sampling:
  - temperature: 0.0
    top_p: 1.0
    max_tokens: 64
prompts:
  - id: "p1"
    text: "Hello"
    constraints:
      - name: "c1"
        description: "must be json"
        kind: "format"
        weight: 1.0
        rules:
          type: "json"
"""
        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write(config)
            path = handle.name

        suite = load_suite(path)
        self.assertEqual(suite.name, "test-suite")
        self.assertEqual(suite.run.concurrency, 4)
        self.assertEqual(suite.run.embeddings.model, "text-embedding-3-small")
        self.assertIn("openai", suite.providers)
        self.assertEqual(len(suite.prompts), 1)
        self.assertEqual(suite.prompts[0].constraints[0].rules["type"], "json")


if __name__ == "__main__":
    unittest.main()
