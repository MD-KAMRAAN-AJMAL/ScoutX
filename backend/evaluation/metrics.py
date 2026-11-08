"""Ranking and paired-comparison metrics with explicit unjudged-item behavior."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Literal

from app.personalization.evaluation import ndcg_at_k, precision_at_k, recall_at_k

UnjudgedPolicy = Literal["irrelevant", "exclude"]


def _deduplicated(ids: Sequence[str]) -> tuple[list[str], int]:
    output: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0
    for value in ids:
        if value in seen:
            duplicate_count += 1
            continue
        seen.add(value)
        output.append(value)
    return output, duplicate_count


def _known_ranked(
    ranked_ids: Sequence[str], relevance: Mapping[str, int], policy: UnjudgedPolicy
) -> list[str]:
    ranked, _ = _deduplicated(ranked_ids)
    if policy == "exclude":
        return [item for item in ranked if item in relevance]
    return ranked


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    policy: UnjudgedPolicy = "irrelevant",
) -> float:
    ranked = _known_ranked(ranked_ids, relevance, policy)
    return next(
        (1.0 / rank for rank, item in enumerate(ranked, 1) if relevance.get(item, 0) > 0),
        0.0,
    )


def average_precision(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    policy: UnjudgedPolicy = "irrelevant",
) -> float:
    ranked = _known_ranked(ranked_ids, relevance, policy)
    relevant_total = sum(grade > 0 for grade in relevance.values())
    if not relevant_total:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranked, 1):
        if relevance.get(item, 0) > 0:
            hits += 1
            score += hits / rank
    return score / relevant_total


def hit_rate_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
    *,
    policy: UnjudgedPolicy = "irrelevant",
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = _known_ranked(ranked_ids, relevance, policy)
    return float(any(relevance.get(item, 0) > 0 for item in ranked[:k]))


def query_metrics(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, int],
    k_values: Sequence[int] = (1, 3, 5, 10),
    *,
    unjudged_policy: UnjudgedPolicy = "irrelevant",
) -> dict[str, float | int]:
    if unjudged_policy not in {"irrelevant", "exclude"}:
        raise ValueError("unjudged_policy must be irrelevant or exclude")
    ranked, duplicate_count = _deduplicated(ranked_ids)
    known = _known_ranked(ranked, relevance, unjudged_policy)
    result: dict[str, float | int] = {
        "mrr": reciprocal_rank(known, relevance, policy=unjudged_policy),
        "map": average_precision(known, relevance, policy=unjudged_policy),
        "zero_result": int(not ranked),
        "duplicate_result_count": duplicate_count,
        "unjudged_result_count": sum(item not in relevance for item in ranked),
    }
    for k in k_values:
        if k < 1:
            raise ValueError("all K values must be positive")
        if unjudged_policy == "exclude":
            selected = known[:k]
            precision = sum(relevance.get(item, 0) > 0 for item in selected) / max(1, len(selected))
            recalled = sum(relevance.get(item, 0) > 0 for item in selected)
            recall = recalled / max(1, sum(grade > 0 for grade in relevance.values()))
        else:
            precision = precision_at_k(
                known, {key for key, grade in relevance.items() if grade > 0}, k
            )
            recall = recall_at_k(known, {key for key, grade in relevance.items() if grade > 0}, k)
        result[f"precision@{k}"] = precision
        result[f"recall@{k}"] = recall
        result[f"hit_rate@{k}"] = hit_rate_at_k(known, relevance, k, policy=unjudged_policy)
        result[f"ndcg@{k}"] = ndcg_at_k(known, relevance, k)
    return result


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def edit_error_rate(
    reference: str, hypothesis: str, *, unit: Literal["character", "word"]
) -> float:
    """Compute standard Levenshtein error rate (CER/WER) against trusted reference text."""
    if unit == "character":
        expected = list("".join(reference.casefold().split()))
        observed = list("".join(hypothesis.casefold().split()))
    elif unit == "word":
        expected = reference.casefold().split()
        observed = hypothesis.casefold().split()
    else:
        raise ValueError("unit must be character or word")
    if not expected:
        raise ValueError("reference text must contain at least one character or word")
    previous = list(range(len(observed) + 1))
    for expected_index, expected_item in enumerate(expected, start=1):
        current = [expected_index]
        for observed_index, observed_item in enumerate(observed, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[observed_index] + 1,
                    previous[observed_index - 1] + (expected_item != observed_item),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def bootstrap_mean_interval(
    values: Sequence[float], *, confidence: float = 0.95, samples: int = 2000, seed: int = 17
) -> dict[str, float | int | None]:
    """Percentile bootstrap interval for a per-query mean; small samples remain descriptive."""
    if not 0 < confidence < 1 or samples < 100:
        raise ValueError("confidence must be in (0, 1) and samples must be at least 100")
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    rng = random.Random(seed)
    raw = [float(value) for value in values]
    means = [sum(rng.choices(raw, k=len(raw))) / len(raw) for _ in range(samples)]
    alpha = (1 - confidence) * 100
    return {
        "n": len(raw),
        "mean": sum(raw) / len(raw),
        "lower": percentile(means, alpha / 2),
        "upper": percentile(means, 100 - alpha / 2),
        "confidence": confidence,
        "method": "nonparametric percentile bootstrap over queries",
        "seed": seed,
        "samples": samples,
    }
