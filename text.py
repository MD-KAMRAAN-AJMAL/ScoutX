"""Text normalization helpers used by parsing, search, and ranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

SEMANTIC_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "budget": ("cost", "expense", "finance", "forecast", "numbers", "pricing"),
    "cost": ("budget", "expense", "finance", "pricing"),
    "deck": ("presentation", "slides", "ppt", "pptx"),
    "numbers": ("budget", "data", "finance", "metrics", "spreadsheet"),
    "pricing": ("budget", "cost", "forecast", "revenue"),
    "revenue": ("pricing", "forecast", "sales"),
    "sheet": ("spreadsheet", "table", "workbook", "xlsx"),
}


def tokenize(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.lower())


def term_counter(value: str | Iterable[str]) -> Counter[str]:
    if isinstance(value, str):
        return Counter(tokenize(value))
    return Counter(value)


def expand_terms(terms: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for value in (term, *SEMANTIC_EXPANSIONS.get(term, ())):
            if value not in seen:
                seen.add(value)
                expanded.append(value)
    return expanded


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)
