"""Compare saved evaluation reports without imposing implicit regression thresholds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evaluation.metrics import bootstrap_mean_interval

Threshold = float | Mapping[str, float]


def compare_reports(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Threshold] | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    if current.get("dataset") != baseline.get("dataset") or current.get(
        "dataset_version"
    ) != baseline.get("dataset_version"):
        raise ValueError("baseline and current reports must use the same dataset and version")
    current_results = current.get("results", {})
    baseline_results = baseline.get("results", {})
    comparisons = {}
    for group in current_results.keys() & baseline_results.keys():
        current_group = current_results[group]
        baseline_group = baseline_results[group]
        if "runs" in current_group or "runs" in baseline_group:
            current_runs = {item.get("run_name"): item for item in current_group.get("runs", [])}
            baseline_runs = {item.get("run_name"): item for item in baseline_group.get("runs", [])}
            matched = {}
            for run_name in current_runs.keys() & baseline_runs.keys():
                matched[str(run_name)] = _compare_pair(
                    current_runs[run_name],
                    baseline_runs[run_name],
                    group=group,
                    label=str(run_name),
                    thresholds=thresholds or {},
                    seed=seed,
                )
            comparisons[group] = {
                "matching_run_names": sorted(matched),
                "runs": matched,
                "status": "measured_difference_only" if matched else "no_matching_configurations",
            }
        else:
            comparisons[group] = _compare_pair(
                current_group,
                baseline_group,
                group=group,
                label=group,
                thresholds=thresholds or {},
                seed=seed,
            )
    return {
        "baseline_run_id": baseline.get("run_id"),
        "current_run_id": current.get("run_id"),
        "comparisons": comparisons,
        "interpretation": "Intervals describe paired query resampling; they are not a universal regression decision. Thresholds are caller-supplied; use min_delta for higher-is-better metrics and max_delta for costs.",
    }


def _compare_pair(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    group: str,
    label: str,
    thresholds: Mapping[str, Threshold],
    seed: int,
) -> dict[str, Any]:
    current_aggregate = current.get("aggregate", {})
    baseline_aggregate = baseline.get("aggregate", {})
    differences = {
        metric: float(current_aggregate[metric]) - float(baseline_aggregate[metric])
        for metric in current_aggregate.keys() & baseline_aggregate.keys()
        if isinstance(current_aggregate[metric], (int, float))
        and isinstance(baseline_aggregate[metric], (int, float))
    }
    current_rows = {item.get("query_id"): item for item in current.get("per_query", [])}
    baseline_rows = {item.get("query_id"): item for item in baseline.get("per_query", [])}
    paired: dict[str, list[float]] = {}
    for query_id in current_rows.keys() & baseline_rows.keys():
        current_metrics = current_rows[query_id].get("metrics", {})
        baseline_metrics = baseline_rows[query_id].get("metrics", {})
        for metric in current_metrics.keys() & baseline_metrics.keys():
            left, right = current_metrics[metric], baseline_metrics[metric]
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                paired.setdefault(metric, []).append(float(left) - float(right))
    intervals = {
        metric: bootstrap_mean_interval(values, seed=seed) for metric, values in paired.items()
    }
    threshold_results = {}
    for metric, delta in differences.items():
        raw = _find_threshold(thresholds, group, label, metric)
        if raw is None:
            continue
        spec = {"min_delta": raw} if isinstance(raw, (int, float)) else raw
        if "min_delta" in spec:
            minimum = float(spec["min_delta"])
            threshold_results[metric] = {
                "delta": delta,
                "min_delta": minimum,
                "threshold_crossed": delta < minimum,
            }
        elif "max_delta" in spec:
            maximum = float(spec["max_delta"])
            threshold_results[metric] = {
                "delta": delta,
                "max_delta": maximum,
                "threshold_crossed": delta > maximum,
            }
    if any(item["threshold_crossed"] for item in threshold_results.values()):
        status = "configured_threshold_crossed"
    elif intervals:
        status = "paired_bootstrap_interval_reported"
    else:
        status = "measured_difference_only"
    return {
        "aggregate_deltas": differences,
        "paired_query_bootstrap_intervals": intervals,
        "configured_thresholds": threshold_results,
        "paired_query_count": len(current_rows.keys() & baseline_rows.keys()),
        "status": status,
    }


def _find_threshold(
    thresholds: Mapping[str, Threshold], group: str, label: str, metric: str
) -> Threshold | None:
    for key in (
        f"{group}.{label}.{metric}",
        f"{group}:{label}:{metric}",
        f"{group}.{metric}",
        metric,
    ):
        if key in thresholds:
            return thresholds[key]
    return None
