"""Command-line entry point for ScoutX."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from scoutx import __version__
from scoutx.document_loader import DocumentLoader
from scoutx.engine import ScoutXEngine
from scoutx.llm import ALLOWED_PROVIDERS, LLMConfig, LLMError, create_llm_provider
from scoutx.storage import UsageLog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutx",
        description="ScoutX agentic file discovery and recommendation system.",
    )
    parser.add_argument("--version", action="store_true", help="print the ScoutX version")
    subparsers = parser.add_subparsers(dest="command")

    search = subparsers.add_parser("search", help="run a ScoutX file recommendation query")
    search.add_argument("query", help="natural-language file discovery query")
    search.add_argument("--root", required=True, help="directory to search")
    search.add_argument("--usage-db", help="SQLite usage/profile database path")
    search.add_argument("--limit", type=int, default=5, help="maximum recommendations to print")
    search.add_argument("--trace", action="store_true", help="print router trace details")
    search.add_argument(
        "--llm-provider",
        choices=sorted(ALLOWED_PROVIDERS),
        help="enable an LLM provider (ollama, openai_compatible, or gemini)",
    )
    search.add_argument(
        "--llm-model",
        help="model identifier for the selected LLM provider (or set SCOUTX_LLM_MODEL)",
    )
    search.add_argument(
        "--llm-base-url",
        help="override the provider HTTP base URL",
    )
    search.add_argument(
        "--llm-api-key-env",
        help="environment variable name containing the provider API key (never pass the secret key directly)",
    )
    search.add_argument(
        "--llm-timeout",
        type=float,
        default=8.0,
        help="positive finite HTTP timeout in seconds for LLM calls (default: 8.0)",
    )
    search.add_argument(
        "--require-llm",
        action="store_true",
        help="require LLM-generated explanations (fail unless --offline-fallback is set)",
    )
    search.add_argument(
        "--offline-fallback",
        action="store_true",
        help="allow explicit degraded deterministic fallback when the LLM is unavailable",
    )

    log_open = subparsers.add_parser("log-open", help="record a file-open event for personalization")
    log_open.add_argument("path", help="file that was opened")
    log_open.add_argument("--usage-db", required=True, help="SQLite usage/profile database path")
    log_open.add_argument("--topic", action="append", default=[], help="topic associated with this open event")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"ScoutX {__version__}")
        return 0

    if args.command == "search":
        return _search(args)
    if args.command == "log-open":
        return _log_open(args)

    parser.print_help()
    return 0


def _search(args: argparse.Namespace) -> int:
    llm_provider = None
    require_llm: bool | None = True if args.require_llm else None

    if args.llm_provider:
        model_name = (args.llm_model or os.environ.get("SCOUTX_LLM_MODEL") or "").strip()
        if not model_name:
            print(
                "Error: --llm-model (or SCOUTX_LLM_MODEL) is required when --llm-provider is specified.",
                file=sys.stderr,
            )
            return 2
        try:
            config = LLMConfig(
                provider=args.llm_provider,
                model=model_name,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
                timeout_seconds=args.llm_timeout,
            )
            llm_provider = create_llm_provider(config)
        except (ValueError, TypeError) as exc:
            print(f"Error: invalid LLM configuration: {exc}", file=sys.stderr)
            return 2

    try:
        engine = ScoutXEngine.from_directory(
            args.root,
            usage_db=args.usage_db,
            llm_provider=llm_provider,
            require_llm=require_llm,
            allow_degraded_fallback=bool(args.offline_fallback),
        )
        response = engine.search(args.query, limit=args.limit)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not response.recommendations:
        print("No recommendations found.")
    for index, recommendation in enumerate(response.recommendations, start=1):
        print(f"{index}. {recommendation.document.path}")
        print(f"   score: {recommendation.score:.3f}")
        print(f"   {recommendation.explanation}")

    if args.trace:
        print("\nRouter trace:")
        for step in response.router_result.trace:
            tools = ", ".join(tool.value for tool in step.strategy.tools)
            print(
                "   "
                f"iteration={step.iteration} "
                f"lane={step.strategy.lane.value} "
                f"tools={tools} "
                f"complexity={step.complexity_score:.3f} "
                f"candidates={step.candidates_seen} "
                f"confident={step.reflection.confident} "
                f"reason={step.reflection.reason}"
            )

    return 0


def _log_open(args: argparse.Namespace) -> int:
    path = Path(args.path)
    document = DocumentLoader().load_file(path)
    if document is None:
        print(f"Could not read file: {path}")
        return 1

    UsageLog(args.usage_db).record_open(document, topics=tuple(args.topic))
    print(f"Recorded open event for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
