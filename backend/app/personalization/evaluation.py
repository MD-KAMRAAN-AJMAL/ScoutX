"""Small offline ranking metrics for controlled recommendation fixtures."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    selected = ranked_ids[:k]
    return sum(item in relevant_ids for item in selected) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0
    return sum(item in relevant_ids for item in ranked_ids[:k]) / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    return next(
        (1.0 / rank for rank, item in enumerate(ranked_ids, start=1) if item in relevant_ids),
        0.0,
    )


def ndcg_at_k(ranked_ids: Sequence[str], relevance: dict[str, float], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    dcg = sum(
        (2 ** max(0.0, relevance.get(item, 0.0)) - 1) / math.log2(rank + 1)
        for rank, item in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted((max(0.0, score) for score in relevance.values()), reverse=True)[:k]
    idcg = sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def category_diversity(categories: Sequence[str], k: int) -> float:
    selected = categories[:k]
    return len(set(selected)) / len(selected) if selected else 0.0
