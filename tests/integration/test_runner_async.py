import tempfile
import unittest

from promptlint.aggregators.robustness import RobustnessAggregator
from promptlint.cache.sqlite_cache import CacheStore
from promptlint.core.runner import SuiteRunner
from promptlint.core.types import ModelOutput, ModelSpec, Prompt, RunConfig, SamplingConfig
from promptlint.config.loader import SuiteSpec
from promptlint.evaluators.robustness import RobustnessEvaluator
from promptlint.ladder.ladder import ModelLadder
from promptlint.providers.base import ModelProvider


class DummyProvider(ModelProvider):
    name = "dummy"

    async def generate_async(self, prompt, model, sampling):
        return ModelOutput(
            text=f"output:{prompt.id}",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


class FailingProvider(ModelProvider):
    name = "dummy"

    async def generate_async(self, prompt, model, sampling):
        raise RuntimeError("should not be called")


class RunnerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_uses_cache(self) -> None:
        prompt = Prompt(id="p1", text="hello")
        model = ModelSpec(provider="dummy", name="dummy-1", tier=1)
        ladder = ModelLadder(name="default", models=[model])
        sampling = [SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=16)]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = f"{temp_dir}/cache.sqlite"
            run = RunConfig(concurrency=1, cache_path=cache_path, use_cache=True)
            suite = SuiteSpec(
                name="test",
                prompts=[prompt],
                ladder=ladder,
                sampling=sampling,
                providers={},
                run=run,
            )

            cache = CacheStore(cache_path)
            runner = SuiteRunner(
                providers={"dummy": DummyProvider()},
                evaluator=RobustnessEvaluator(),
                aggregator=RobustnessAggregator(),
                cache=cache,
                concurrency=1,
                show_progress=False,
            )
            first = await runner.run(suite)
            self.assertFalse(first.prompt_results[0].runs[0].metadata.get("cached"))
            cache.close()

            cache = CacheStore(cache_path)
            runner = SuiteRunner(
                providers={"dummy": FailingProvider()},
                evaluator=RobustnessEvaluator(),
                aggregator=RobustnessAggregator(),
                cache=cache,
                concurrency=1,
                show_progress=False,
            )
            second = await runner.run(suite)
            self.assertTrue(second.prompt_results[0].runs[0].metadata.get("cached"))
            cache.close()


if __name__ == "__main__":
    unittest.main()
