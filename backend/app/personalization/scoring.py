"""Bounded ranking adjustments that keep retrieval/reranker scores primary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.personalization.profile import UserProfile

MAX_PERSONALIZATION_ADJUSTMENT = 0.08
MIN_PERSONALIZATION_RELEVANCE = 0.20


@dataclass(frozen=True)
class PersonalizedScore:
    score: float
    adjustment: float
    reasons: tuple[str, ...]
    excluded: bool = False


def score_file(
    *,
    base_score: float,
    file_id: str,
    extension: str | None,
    root_id: str | None,
    profile: UserProfile,
    personalization_enabled: bool = True,
) -> PersonalizedScore:
    """Apply a transparent bounded score delta; the base rank remains dominant.

    Formula: base + up to 0.025 for a preferred extension or root, 0.015 for
    decayed type/root feedback, 0.03 * decayed access weight, and 0.06 * latest
    explicit file-feedback weight.
    The combined adjustment is clamped to [-0.07, +0.08], and only applies to
    retrieval candidates whose reranker score is at least 0.20.
    """
    prefs = profile.preferences
    normalized_extension = (extension or "").lower()
    excluded_extensions = {str(value).lower() for value in prefs.get("excluded_extensions", [])}
    excluded_roots = {str(value) for value in prefs.get("excluded_root_ids", [])}
    if normalized_extension in excluded_extensions or (root_id and root_id in excluded_roots):
        return PersonalizedScore(0.0, 0.0, (), excluded=True)
    base = min(1.0, max(0.0, float(base_score)))
    if not personalization_enabled or not profile.enabled or base < MIN_PERSONALIZATION_RELEVANCE:
        return PersonalizedScore(base, 0.0, ())

    parts: list[float] = []
    reasons: list[str] = []
    preferred_extensions = {item.lower() for item in profile.preferred_extensions}
    if normalized_extension and normalized_extension in preferred_extensions:
        parts.append(0.025)
        reasons.append("Matches a file type you prefer")
    extension_feedback = min(
        1.0, max(-1.0, profile.extension_feedback.get(normalized_extension, 0.0))
    )
    if extension_feedback > 0:
        parts.append(0.015 * extension_feedback)
        reasons.append("Related to file types you marked useful")
    elif extension_feedback < 0:
        parts.append(0.015 * extension_feedback)
        reasons.append("Reduced after negative feedback on similar file types")
    preferred_roots = set(profile.frequent_root_ids) | {
        str(value) for value in prefs.get("preferred_root_ids", [])
    }
    if root_id and root_id in preferred_roots:
        parts.append(0.025)
        reasons.append("From a folder you often use")
    root_feedback = min(1.0, max(-1.0, profile.root_feedback.get(root_id or "", 0.0)))
    if root_feedback > 0:
        parts.append(0.015 * root_feedback)
        reasons.append("Related to files from a folder you marked useful")
    elif root_feedback < 0:
        parts.append(0.015 * root_feedback)
        reasons.append("Reduced after negative feedback on a similar folder")
    access = min(1.0, max(0.0, profile.file_access_weights.get(file_id, 0.0)))
    if access > 0:
        parts.append(0.03 * access)
        reasons.append("Recently accessed")
    feedback = min(1.0, max(-1.0, profile.file_feedback.get(file_id, 0.0)))
    if feedback > 0:
        parts.append(0.06 * feedback)
        reasons.append("You marked this file useful")
    elif feedback < 0:
        parts.append(0.06 * feedback)
        reasons.append("Reduced after your negative feedback")
    adjustment = min(MAX_PERSONALIZATION_ADJUSTMENT, max(-0.07, sum(parts)))
    return PersonalizedScore(
        score=min(1.0, max(0.0, base + adjustment)),
        adjustment=adjustment,
        reasons=tuple(reasons),
    )


def personalize_ranked_candidates(
    candidates: list[dict[str, Any]],
    profile: UserProfile,
    *,
    enabled: bool = True,
    file_root_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return scoped, de-duplicated candidate records in adjusted score order."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for candidate in candidates:
        candidate_file = candidate.get("candidate", candidate)
        file_id = str(candidate_file.get("file_id", ""))
        chunk_id = candidate_file.get("chunk_id")
        identity = (file_id, str(chunk_id) if chunk_id else None)
        if not file_id or identity in seen:
            continue
        seen.add(identity)
        score = score_file(
            base_score=float(
                candidate.get("final_score", candidate_file.get("relevance_score", 0))
            ),
            file_id=file_id,
            extension=candidate_file.get("extension"),
            root_id=(
                str((file_root_ids or {}).get(file_id) or candidate_file.get("root_id"))
                if (file_root_ids or {}).get(file_id) or candidate_file.get("root_id")
                else None
            ),
            profile=profile,
            personalization_enabled=enabled,
        )
        if score.excluded:
            continue
        updated = dict(candidate)
        updated["final_score"] = score.score
        updated["personalization_adjustment"] = score.adjustment
        updated["personalization_reasons"] = list(score.reasons)
        if "candidate" in updated:
            nested = dict(candidate_file)
            nested["relevance_score"] = score.score
            updated["candidate"] = nested
        else:
            updated["relevance_score"] = score.score
        result.append(updated)
    return sorted(result, key=lambda item: float(item.get("final_score", 0)), reverse=True)
