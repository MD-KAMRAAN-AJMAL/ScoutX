"""Command-line entry point for ScoutX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scoutx import __version__
from scoutx.document_loader import DocumentLoader
from scoutx.engine import ScoutXEngine
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
    engine = ScoutXEngine.from_directory(args.root, usage_db=args.usage_db)
    response = engine.search(args.query, limit=args.limit)

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
