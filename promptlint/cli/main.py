from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

from promptlint.aggregators.robustness import RobustnessAggregator
from promptlint.cache.sqlite_cache import CacheStore
from promptlint.config.loader import load_suite
from promptlint.core.runner import SuiteRunner
from promptlint.embeddings.registry import build_embedder
from promptlint.evaluators import RobustnessEvaluator
from promptlint.providers.registry import build_providers, close_providers
from promptlint.reporting import HtmlReport, JsonReport, MarkdownReport
from promptlint.reporting.serializer import suite_result_to_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptlint",
        description="Evaluate prompt robustness across model ladders.",
    )
    parser.add_argument("--suite", required=True, help="Path to suite.yaml")
    parser.add_argument(
        "--report",
        default="report.md",
        help="Output path for report markdown",
    )
    parser.add_argument(
        "--report-format",
        default="markdown",
        choices=["markdown", "json", "html"],
        help="Report format",
    )
    parser.add_argument(
        "--runs-output",
        default="",
        help="Optional path to write full run details as JSON",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw provider payload in runs output",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    suite_path = Path(args.suite)
    if not suite_path.exists():
        print(f"Suite file not found: {suite_path}", file=sys.stderr)
        return 1

    try:
        return asyncio.run(_run_async(args))
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 2


async def _run_async(args: argparse.Namespace) -> int:
    suite = load_suite(str(args.suite))
    if not suite.providers:
        print(
            "No providers configured. Add a providers section in the suite.",
            file=sys.stderr,
        )
        return 1

    providers = build_providers(suite.providers)
    cache = None
    embedder = None
    try:
        if suite.run.cache_path and suite.run.use_cache:
            cache = CacheStore(suite.run.cache_path)
        embedder = build_embedder(suite.providers, suite.run.embeddings)

        evaluator = RobustnessEvaluator()
        aggregator = RobustnessAggregator()
        runner = SuiteRunner(
            providers=providers,
            evaluator=evaluator,
            aggregator=aggregator,
            cache=cache,
            embedder=embedder,
            concurrency=suite.run.concurrency,
        )

        result = await runner.run(suite)
        scores = [prompt_result.score for prompt_result in result.prompt_results]

        if args.report_format == "html":
            renderer = HtmlReport()
        elif args.report_format == "json":
            renderer = JsonReport()
        else:
            renderer = MarkdownReport()
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(renderer.render(scores), encoding="utf-8")

        if args.runs_output:
            runs_path = Path(args.runs_output)
            runs_path.parent.mkdir(parents=True, exist_ok=True)
            runs_path.write_text(
                suite_result_to_json(result, include_raw=args.include_raw),
                encoding="utf-8",
            )

        print(f"Completed suite: {suite.name} ({len(suite.prompts)} prompts)")
        print(f"Report written to: {report_path}")
        if args.runs_output:
            print(f"Run details written to: {args.runs_output}")
        return 0
    finally:
        if embedder:
            await embedder.aclose()
        await close_providers(providers)
        if cache:
            cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
