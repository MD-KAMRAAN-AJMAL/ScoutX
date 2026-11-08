"""Reproducible report serialization and environment capture."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any


def environment_snapshot() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    try:
        import importlib.metadata

        package_versions = {
            name: importlib.metadata.version(name)
            for name in ("qdrant-client", "sqlalchemy", "fastapi", "langgraph")
        }
    except PackageNotFoundError:
        package_versions = {}
    total_memory = None
    try:
        total_memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "python": sys.version.split()[0],
        "package_versions": package_versions,
        "git_commit": sha,
        "accelerator": "not probed; model benchmarks must record device in provider configuration",
    }


def new_report(*, dataset: str, dataset_version: str, mode: str, seed: int) -> dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "dataset_version": dataset_version,
        "mode": mode,
        "seed": seed,
        "environment": environment_snapshot(),
        "results": {},
        "not_run": {},
        "limitations": [],
    }


def write_reports(report: dict[str, Any], output_dir: str | Path, formats: set[str]) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_paths = []
    if "json" in formats:
        path = directory / f"{report['run_id']}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        output_paths.append(path)
    if "markdown" in formats:
        path = directory / f"{report['run_id']}.md"
        path.write_text(_markdown(report), encoding="utf-8")
        output_paths.append(path)
    if "csv" in formats:
        path = directory / f"{report['run_id']}.csv"
        _write_csv(report, path)
        output_paths.append(path)
    return output_paths


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# ScoutX evaluation: {report['mode']}",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Dataset: `{report['dataset']}@{report['dataset_version']}`",
        f"- Timestamp (UTC): {report['timestamp_utc']}",
        f"- Git commit: `{report['environment'].get('git_commit') or 'unknown'}`",
        "",
    ]
    for group, result in report["results"].items():
        lines.extend([f"## {group}", ""])
        aggregate = result.get("aggregate", {})
        if aggregate:
            lines.extend(["| Metric | Value |", "|---|---:|"])
            lines.extend(f"| {key} | {value} |" for key, value in aggregate.items())
            lines.append("")
        if "runs" in result:
            lines.extend(["| Configuration | Metrics |", "|---|---|"])
            for run in result["runs"]:
                lines.append(
                    f"| {run.get('run_name')} | `{json.dumps(run.get('aggregate', {}), sort_keys=True)}` |"
                )
            lines.append("")
    if report["not_run"]:
        lines.extend(["## Not run", ""])
        lines.extend(
            f"- **{key}: NOT RUN** — {reason}" for key, reason in report["not_run"].items()
        )
    if report["limitations"]:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for group, result in report["results"].items():
        if "per_query" in result:
            rows.extend({"group": group, **item} for item in result["per_query"])
        elif "per_scenario" in result:
            rows.extend({"group": group, **item} for item in result["per_scenario"])
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["group"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
