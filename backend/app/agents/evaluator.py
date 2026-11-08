"""Retrieval sufficiency evaluator and Phase 7 reranking interface."""

import re
from typing import Any

from app.agents.schemas import QueryUnderstanding, RelevanceEvaluation

_STOP = {
    "a",
    "an",
    "and",
    "about",
    "find",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def evaluate_candidates(
    query: str, intent: QueryUnderstanding, candidates: list[dict[str, Any]]
) -> RelevanceEvaluation:
    count = len(candidates)
    terms = [x.lower() for x in re.findall(r"[\w.+-]+", query) if x.lower() not in _STOP]
    if not candidates:
        terminal = intent.intent in {"metadata", "exact_filename"}
        return RelevanceEvaluation(
            sufficient=terminal,
            candidate_count=0,
            lexical_coverage=0,
            reason="No matches." if terminal else "No candidates; try one adaptive pass.",
        )
    coverage = max(
        (
            sum(
                term in (str(x.get("file_name", "")) + " " + str(x.get("content", ""))).lower()
                for term in terms
            )
            / max(1, len(terms))
            for x in candidates
        ),
        default=0,
    )
    sufficient = count >= 3 or coverage >= 0.35
    return RelevanceEvaluation(
        sufficient=sufficient,
        candidate_count=count,
        lexical_coverage=round(coverage, 4),
        reason=f"{count} candidates; best lexical coverage {coverage:.2f}.",
    )
