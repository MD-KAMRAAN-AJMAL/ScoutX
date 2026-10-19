"""Command-line interface for ScoutX."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scoutx import __version__
from scoutx.document_loader import DocumentLoader
from scoutx.embeddings import (
    ALLOWED_EMBEDDING_PROVIDERS,
    EmbeddingConfig,
    EmbeddingError,
    create_embedding_provider,
)
from scoutx.engine import ScoutXEngine
from scoutx.laya import LayaConfig, LayaHTTPAdvisor
from scoutx.llm import LLMConfig, LLMError, create_llm_provider
from scoutx.storage import StorageSchemaError, UsageLog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutx",
        description="ScoutX agentic file discovery and recommendation system.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="show program's version number and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="available commands")

    search = subparsers.add_parser("search", help="search for documents matching a query")
    search.add_argument("query", help="natural-language search query")
    search.add_argument(
        "--root",
        default=os.getcwd(),
        help="root directory to search (default: current directory)",
    )
    search.add_argument(
        "--limit",
        type=int,
        default=5,
        help="maximum number of recommendations to return (default: 5)",
    )
    search.add_argument(
        "--trace",
        action="store_true",
        help="print internal routing loop trace to stderr",
    )
    search.add_argument(
        "--usage-db",
        help="SQLite database path for persisting query history and personalization",
    )
    search.add_argument(
        "--index-db",
        help="dedicated SQLite database path for persistent metadata and vector indices",
    )
    search.add_argument(
        "--llm-provider",
        choices=["ollama", "openai_compatible", "gemini"],
        help="LLM provider for grounded explanations and query refinement",
    )
    search.add_argument(
        "--llm-model",
        help="model identifier for the selected LLM provider (or set SCOUTX_LLM_MODEL)",
    )
    search.add_argument(
        "--llm-base-url",
        help="override the LLM provider HTTP base URL",
    )
    search.add_argument(
        "--llm-api-key-env",
        help="environment variable name containing the LLM provider API key",
    )
    search.add_argument(
        "--llm-timeout",
        type=float,
        default=3.0,
        help="positive finite HTTP timeout in seconds for LLM calls (default: 3.0)",
    )
    search.add_argument(
        "--require-llm",
        action="store_true",
        help="fail fast if the requested LLM provider is unreachable",
    )
    search.add_argument(
        "--offline-fallback",
        action="store_true",
        help="allow deterministic offline fallback explanation if LLM fails (degraded mode)",
    )
    search.add_argument(
        "--embedding-provider",
        choices=sorted(ALLOWED_EMBEDDING_PROVIDERS),
        help="enable an embedding provider (local_hash, ollama, openai_compatible, or gemini)",
    )
    search.add_argument(
        "--embedding-model",
        help="model identifier for the selected embedding provider (or set SCOUTX_EMBEDDING_MODEL)",
    )
    search.add_argument(
        "--embedding-dimensions",
        type=int,
        help="embedding vector dimensions (defaults to 64 for local_hash)",
    )
    search.add_argument(
        "--embedding-base-url",
        help="override the embedding provider HTTP base URL",
    )
    search.add_argument(
        "--embedding-api-key-env",
        help="environment variable name containing the embedding provider API key",
    )
    search.add_argument(
        "--embedding-timeout",
        type=float,
        default=5.0,
        help="positive finite HTTP timeout in seconds for embedding calls (default: 5.0)",
    )
    search.add_argument(
        "--laya-shadow",
        action="store_true",
        help="enable Laya System-1 shadow advisory (pure telemetry, never changes routing)",
    )
    search.add_argument(
        "--laya-base-url",
        default="http://127.0.0.1:8001",
        help="Laya HTTP bridge base URL (default: http://127.0.0.1:8001)",
    )
    search.add_argument(
        "--laya-api-key-env",
        help="environment variable name containing the Laya API key/token if authentication is required",
    )
    search.add_argument(
        "--laya-timeout",
        type=float,
        default=0.5,
        help="positive finite client deadline in seconds for Laya advisory HTTP call (default: 0.5s)",
    )

    log_open = subparsers.add_parser(
        "log-open", help="record a file-open event for personalization"
    )
    log_open.add_argument("path", help="file that was opened")
    log_open.add_argument("--usage-db", required=True, help="SQLite usage/profile database path")
    log_open.add_argument(
        "--topic", action="append", default=[], help="topic associated with this open event"
    )

    launch = subparsers.add_parser(
        "launch", help="open the ScoutX Spotlight-style desktop launcher"
    )
    launch.add_argument("--root", help="root directory to search (defaults to settings search_root)")
    launch.add_argument("--config", help="custom path to config.json settings file")
    launch.add_argument("--usage-db", help="path to usage.db SQLite file")
    launch.add_argument("--index-db", help="path to index.db SQLite file")
    launch.add_argument("--limit", type=int, default=None, help="maximum results to display")

    gui = subparsers.add_parser(
        "gui", help="alias for launch: open the ScoutX Spotlight-style desktop launcher"
    )
    gui.add_argument("--root", help="root directory to search (defaults to settings search_root)")
    gui.add_argument("--config", help="custom path to config.json settings file")
    gui.add_argument("--usage-db", help="path to usage.db SQLite file")
    gui.add_argument("--index-db", help="path to index.db SQLite file")
    gui.add_argument("--limit", type=int, default=None, help="maximum results to display")

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
    if args.command in {"launch", "gui"}:
        return _launch_gui(args)

    parser.print_help()
    return 0


def _launch_gui(args: argparse.Namespace) -> int:
    try:
        from scoutx.launcher import run_launcher
        return run_launcher(args)
    except ImportError as exc:
        print(
            "Error: GUI dependencies not installed.\n"
            "Install with: pip install 'scoutx[gui]' (or pip install PySide6)",
            file=sys.stderr,
        )
        return 2


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

    embedding_provider = None
    if args.embedding_provider:
        default_emb_model = (
            "scoutx-feature-hash-v1" if args.embedding_provider == "local_hash" else ""
        )
        emb_model_name = (
            args.embedding_model
            or os.environ.get("SCOUTX_EMBEDDING_MODEL")
            or default_emb_model
        ).strip()
        if not emb_model_name:
            print(
                "Error: --embedding-model (or SCOUTX_EMBEDDING_MODEL) is required when --embedding-provider is specified.",
                file=sys.stderr,
            )
            return 2
        try:
            emb_config = EmbeddingConfig(
                provider=args.embedding_provider,
                model=emb_model_name,
                dimensions=args.embedding_dimensions,
                base_url=args.embedding_base_url,
                api_key_env=args.embedding_api_key_env,
                timeout_seconds=args.embedding_timeout,
            )
            embedding_provider = create_embedding_provider(emb_config)
        except (ValueError, TypeError) as exc:
            print(f"Error: invalid embedding configuration: {exc}", file=sys.stderr)
            return 2

    laya_advisor = None
    if getattr(args, "laya_shadow", False):
        try:
            laya_cfg = LayaConfig(
                enabled=True,
                base_url=args.laya_base_url,
                api_key_env=args.laya_api_key_env,
                timeout_seconds=args.laya_timeout,
            )
            laya_advisor = LayaHTTPAdvisor(laya_cfg)
        except (ValueError, TypeError) as exc:
            print(f"Error: invalid Laya configuration: {exc}", file=sys.stderr)
            return 2

    try:
        engine = ScoutXEngine.from_directory(
            args.root,
            usage_db=args.usage_db,
            index_db=args.index_db,
            loader=DocumentLoader(),
            llm_provider=llm_provider,
            require_llm=require_llm,
            allow_degraded_fallback=bool(args.offline_fallback),
            embedding_provider=embedding_provider,
            laya_advisor=laya_advisor,
        )
        response = engine.search(args.query, limit=args.limit)
    except (LLMError, EmbeddingError, StorageSchemaError) as exc:
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
            trace_str = (
                "   "
                f"iteration={step.iteration} "
                f"lane={step.strategy.lane.value} "
                f"tools={tools} "
                f"complexity={step.complexity_score:.3f} "
                f"candidates={step.candidates_seen} "
                f"confident={step.reflection.confident} "
                f"reason={step.reflection.reason}"
            )
            if step.laya_advisory is not None:
                adv = step.laya_advisory
                trace_str += (
                    f" | laya[status={adv.status} "
                    f"suggested_lane={adv.suggested_lane.value if adv.suggested_lane else 'none'} "
                    f"escalate_prob={adv.escalation_probability if adv.escalation_probability is not None else 'none'} "
                    f"conf={adv.confidence if adv.confidence is not None else 'none'}]"
                )
            print(trace_str)

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
