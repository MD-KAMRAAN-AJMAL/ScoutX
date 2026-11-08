"""Locally runnable evaluation CLI; model-backed output is supplied as explicit artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.compare import compare_reports
from evaluation.dataset import DEFAULT_DATASET, load_dataset
from evaluation.metrics import bootstrap_mean_interval
from evaluation.privacy import safe_configuration
from evaluation.report import new_report, write_reports
from evaluation.runner import (
    evaluate_agent,
    evaluate_citations,
    evaluate_observations,
    evaluate_rag,
    evaluate_rankings,
    evaluate_recommendations,
    evaluate_reranking,
    offline_agent_routing_run,
    offline_lexical_run,
    offline_personalization_runs,
)
from evaluation.schema import EvaluationArtifacts

MODES = (
    "retrieval",
    "reranking",
    "rag",
    "citations",
    "agent",
    "personalization",
    "recommendations",
    "ingestion",
    "extraction",
    "performance",
    "reliability",
    "all",
)


def _provider(spec: str, dataset, mode: str, options: dict[str, Any]):
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("provider must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    return function(dataset=dataset, mode=mode, options=options)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset)
    if args.dataset_version and dataset.version != args.dataset_version:
        raise ValueError(
            f"selected dataset is version {dataset.version}, not requested {args.dataset_version}"
        )
    if any(value < 1 for value in args.k):
        raise ValueError("every K value must be positive")
    if args.repetitions < 1 or args.warmups < 0 or not 1 <= args.concurrency <= 32:
        raise ValueError("repetitions, warmups, or concurrency is outside its supported range")
    extra_configuration = {}
    if args.configuration:
        extra_configuration = json.loads(Path(args.configuration).read_text(encoding="utf-8"))
        if not isinstance(extra_configuration, dict):
            raise ValueError("configuration file must contain a JSON object")
    artifacts = None
    if args.artifacts:
        payload = json.loads(Path(args.artifacts).read_text(encoding="utf-8"))
        artifacts = EvaluationArtifacts.model_validate(payload)
        if artifacts.dataset != dataset.name or artifacts.dataset_version != dataset.version:
            raise ValueError("artifact dataset name/version must match the selected dataset")
    if args.provider:
        produced = _provider(
            args.provider,
            dataset,
            args.mode,
            {
                "k_values": args.k,
                "repetitions": args.repetitions,
                "warmups": args.warmups,
                "concurrency": args.concurrency,
                "seed": args.seed,
                "configuration": extra_configuration,
            },
        )
        artifacts = EvaluationArtifacts.model_validate(produced)
        if artifacts.dataset != dataset.name or artifacts.dataset_version != dataset.version:
            raise ValueError("provider output dataset name/version must match selected dataset")

    report = new_report(
        dataset=dataset.name, dataset_version=dataset.version, mode=args.mode, seed=args.seed
    )
    report["configuration"] = safe_configuration(
        {
            "k_values": args.k,
            "unjudged_policy": args.unjudged,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "concurrency": args.concurrency,
            "provider": args.provider or "offline/default",
            **extra_configuration,
        }
    )
    requested = set(MODES[:-1]) if args.mode == "all" else {args.mode}
    results = report["results"]
    not_run = report["not_run"]
    if "retrieval" in requested:
        runs = [offline_lexical_run(dataset, repetitions=args.repetitions, warmups=args.warmups)]
        if artifacts:
            runs.extend(artifacts.rankings)
        results["retrieval"] = {
            "runs": [
                evaluate_rankings(
                    dataset, item, k_values=args.k, unjudged_policy=args.unjudged, seed=args.seed
                )
                for item in runs
            ],
            "dataset_query_count": len(dataset.queries),
            "offline_baseline_notice": "offline_lexical_token_overlap is a deterministic synthetic-corpus baseline, not ScoutX production search.",
        }

    group_specs = {
        "reranking": (
            artifacts.reranking if artifacts else [],
            lambda rows: evaluate_reranking(dataset, rows, args.k),
        ),
        "rag": (artifacts.rag if artifacts else [], lambda rows: evaluate_rag(dataset, rows)),
        "citations": (
            artifacts.citations if artifacts else [],
            lambda rows: evaluate_citations(dataset, rows),
        ),
        "agent": (artifacts.agent if artifacts else [], lambda rows: evaluate_agent(dataset, rows)),
        "recommendations": (
            artifacts.recommendations if artifacts else [],
            lambda rows: evaluate_recommendations(dataset, rows, args.k),
        ),
        "ingestion": (
            artifacts.ingestion if artifacts else [],
            lambda rows: evaluate_observations(rows, "ingestion"),
        ),
        "extraction": (
            artifacts.ingestion if artifacts else [],
            lambda rows: evaluate_observations(rows, "extraction"),
        ),
        "performance": (
            artifacts.performance if artifacts else [],
            lambda rows: evaluate_observations(rows, "performance"),
        ),
        "reliability": (
            artifacts.reliability if artifacts else [],
            lambda rows: evaluate_observations(rows, "reliability"),
        ),
    }
    for group in requested - {"retrieval", "personalization"}:
        if group == "agent" and not (artifacts and artifacts.agent):
            results[group] = offline_agent_routing_run(dataset)
            not_run["agent_graph"] = (
                "Only the deterministic planner was measured; graph execution and tool calls require a provider or captured artifacts."
            )
            continue
        rows, evaluator = group_specs[group]
        if rows:
            results[group] = evaluator(rows)
        else:
            not_run[group] = (
                "No observations supplied. Run a provider or pass a matching --artifacts file; no live model/service benchmark was fabricated."
            )
    if "personalization" in requested:
        if artifacts and artifacts.personalization:
            personalized = artifacts.personalization
            base = {run.name: run for run in artifacts.rankings}
        else:
            personalized = offline_personalization_runs(dataset)
            offline_base = offline_lexical_run(dataset)
            base = {offline_base.name: offline_base}
        if personalized:
            scores = [
                evaluate_rankings(
                    dataset, run, k_values=args.k, unjudged_policy=args.unjudged, seed=args.seed
                )
                for run in personalized
            ]
            results["personalization"] = {
                "runs": scores,
                "comparison": _paired_personalization(base, personalized, dataset, args.k),
            }
        else:
            not_run["personalization"] = "No personalized ranking observations supplied."
    report["limitations"].extend(
        [
            "Dataset judgments are small and synthetic-authored; do not generalize scores to user libraries.",
            "LLM-based faithfulness and answer quality are not inferred from token overlap; Ragas/model judging is optional and not run by this deterministic CLI.",
            "Latency values from the local token-overlap baseline exclude database, embedding, Qdrant, reranker, and LLM work.",
        ]
    )
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(
            report, baseline, thresholds=_load_thresholds(args.thresholds), seed=args.seed
        )
    return report


def _paired_personalization(base_runs, personalized_runs, dataset, k_values):
    base = next(iter(base_runs.values()), None)
    if base is None:
        return {
            "status": "not_available",
            "reason": "No paired non-personalized ranking was supplied.",
        }
    baseline = evaluate_rankings(dataset, base, k_values=k_values)
    quality_metrics = {
        "mrr",
        "map",
        *(f"precision@{k}" for k in k_values),
        *(f"recall@{k}" for k in k_values),
        *(f"ndcg@{k}" for k in k_values),
        *(f"hit_rate@{k}" for k in k_values),
    }
    baseline_rows = {item["query_id"]: item for item in baseline["per_query"]}
    result = []
    for run in personalized_runs:
        personalized = evaluate_rankings(dataset, run, k_values=k_values)
        personalized_rows = {item["query_id"]: item for item in personalized["per_query"]}
        shared_ids = baseline_rows.keys() & personalized_rows.keys()
        paired_deltas = {
            metric: [
                float(personalized_rows[query_id]["metrics"][metric])
                - float(baseline_rows[query_id]["metrics"][metric])
                for query_id in shared_ids
            ]
            for metric in quality_metrics
        }
        result.append(
            {
                "run_name": run.name,
                "base_run": base.name,
                "metric_deltas": {
                    metric: sum(values) / len(values) if values else None
                    for metric, values in paired_deltas.items()
                },
                "paired_query_intervals": {
                    metric: bootstrap_mean_interval(values)
                    for metric, values in paired_deltas.items()
                },
                "paired_query_count": len(shared_ids),
                "mean_latency_delta_ms": (
                    personalized["aggregate"]["mean_latency_ms"]
                    - baseline["aggregate"]["mean_latency_ms"]
                    if personalized["aggregate"]["mean_latency_ms"] is not None
                    and baseline["aggregate"]["mean_latency_ms"] is not None
                    else None
                ),
            }
        )
    return {
        "comparisons": result,
        "note": "Paired query sets; score changes are measured, not a claim of improvement.",
    }


def _load_thresholds(path: str | None) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=MODES, default="retrieval")
    result.add_argument("--dataset", default=str(DEFAULT_DATASET))
    result.add_argument("--dataset-version", help="Require an exact dataset version")
    result.add_argument(
        "--artifacts", help="JSON captured provider outputs from the matching dataset version"
    )
    result.add_argument("--provider", help="Optional Python provider in module:function form")
    result.add_argument(
        "--configuration", help="JSON file with model and pipeline identifiers/settings"
    )
    result.add_argument("--output-dir", default="evaluation-results")
    result.add_argument("--format", choices=("json", "csv", "markdown", "all"), default="all")
    result.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    result.add_argument("--unjudged", choices=("irrelevant", "exclude"), default="irrelevant")
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--warmups", type=int, default=0)
    result.add_argument("--concurrency", type=int, default=1)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--baseline", help="Previously saved JSON report for comparison")
    result.add_argument("--thresholds", help="JSON map of metric to minimum acceptable delta")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        report = run(args)
        formats = {"json", "csv", "markdown"} if args.format == "all" else {args.format}
        paths = write_reports(report, args.output_dir, formats)
        for path in paths:
            print(path)
        if args.mode != "all" and args.mode != "retrieval" and args.mode not in report["results"]:
            print(f"{args.mode} was NOT RUN; see report.", file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, ImportError, KeyError) as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
