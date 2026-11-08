"""Privacy-conscious, time-decayed user signals derived from existing records."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import File, FileActivity, RecommendationFeedback
from app.database.repositories import PreferenceRepository

ACTIVITY_EVENTS = {
    "opened": 1.0,
    "selected": 1.2,
    "file_opened": 1.0,
    "search_result_opened": 1.0,
    "recommendation_clicked": 1.1,
}
POSITIVE_FEEDBACK = {"relevant", "helpful", "useful"}
NEGATIVE_FEEDBACK = {"not_relevant", "not_helpful", "not_useful", "dismissed"}
DEFAULT_PREFERENCES: dict[str, Any] = {
    "personalization_enabled": True,
    "preferred_extensions": [],
    "excluded_extensions": [],
    "preferred_root_ids": [],
    "excluded_root_ids": [],
    "recommendation_limit": 12,
}
_PROFILE_CACHE_TTL_SECONDS = 30.0
_PROFILE_CACHE: dict[uuid.UUID, tuple[float, UserProfile]] = {}
_PROFILE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class UserProfile:
    """A request-local derived profile; raw activity text is never included."""

    preferences: dict[str, Any] = field(default_factory=dict)
    preferred_extensions: tuple[str, ...] = ()
    frequent_root_ids: tuple[str, ...] = ()
    recent_file_ids: tuple[str, ...] = ()
    file_access_weights: dict[str, float] = field(default_factory=dict)
    file_access_counts: dict[str, int] = field(default_factory=dict)
    file_feedback: dict[str, float] = field(default_factory=dict)
    extension_feedback: dict[str, float] = field(default_factory=dict)
    root_feedback: dict[str, float] = field(default_factory=dict)
    dismissed_file_ids: tuple[str, ...] = ()
    profile_status: str = "available"

    @property
    def enabled(self) -> bool:
        return bool(self.preferences.get("personalization_enabled", True))


def _aware(value: datetime, now: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=now.tzinfo or UTC)
    return value


def _decay(timestamp: datetime, now: datetime, half_life_days: float) -> float:
    age_days = max(0.0, (now - _aware(timestamp, now)).total_seconds() / 86400)
    return math.exp(-math.log(2) * age_days / half_life_days)


def build_user_profile(
    session: Session,
    user_id: uuid.UUID,
    *,
    now: datetime | None = None,
    history_days: int = 180,
    activity_limit: int = 1000,
) -> UserProfile:
    """Derive recent file/type/root preferences from scoped activity and feedback."""
    now = now or datetime.now(UTC)
    preferences_record = PreferenceRepository(session).get(user_id)
    stored_preferences = preferences_record.preferences if preferences_record else {}
    preferences = {**DEFAULT_PREFERENCES, **stored_preferences}
    cutoff = now - timedelta(days=history_days)
    reset_at = preferences.get("profile_reset_at")
    if isinstance(reset_at, str):
        try:
            reset_at = datetime.fromisoformat(reset_at)
        except ValueError:
            reset_at = None
    if isinstance(reset_at, datetime):
        cutoff = max(cutoff, _aware(reset_at, now))

    activity_rows = session.execute(
        select(FileActivity, File.extension, File.root_id)
        .join(File, File.id == FileActivity.file_id)
        .where(
            FileActivity.user_id == user_id,
            File.user_id == user_id,
            FileActivity.created_at >= cutoff,
            FileActivity.event_type.in_(tuple(ACTIVITY_EVENTS)),
        )
        .order_by(FileActivity.created_at.desc())
        .limit(activity_limit)
    ).all()

    access_weights: defaultdict[str, float] = defaultdict(float)
    access_counts: Counter[str] = Counter()
    extension_weights: defaultdict[str, float] = defaultdict(float)
    root_weights: defaultdict[str, float] = defaultdict(float)
    recent_files: list[str] = []
    seen_recent: set[str] = set()
    for event, extension, root_id in activity_rows:
        if event.file_id is None:
            continue
        file_id = str(event.file_id)
        weight = ACTIVITY_EVENTS.get(event.event_type, 0.0) * _decay(event.created_at, now, 30.0)
        access_weights[file_id] += weight
        access_counts[file_id] += 1
        if extension:
            extension_weights[extension.lower()] += weight
        if root_id:
            root_weights[str(root_id)] += weight
        if file_id not in seen_recent:
            recent_files.append(file_id)
            seen_recent.add(file_id)

    feedback_rows = session.execute(
        select(RecommendationFeedback, File.extension, File.root_id)
        .join(
            File,
            (File.id == RecommendationFeedback.file_id)
            & (File.user_id == RecommendationFeedback.user_id),
        )
        .where(
            RecommendationFeedback.user_id == user_id,
            File.user_id == user_id,
            RecommendationFeedback.created_at >= cutoff,
        )
        .order_by(RecommendationFeedback.created_at.desc())
        .limit(activity_limit)
    ).all()
    file_feedback: dict[str, float] = {}
    extension_feedback_weights: defaultdict[str, float] = defaultdict(float)
    root_feedback_weights: defaultdict[str, float] = defaultdict(float)
    dismissed: list[str] = []
    seen_feedback: set[str] = set()
    for item, extension, root_id in feedback_rows:
        file_id = str(item.file_id)
        if file_id in seen_feedback:
            continue
        action = (item.action or "").lower()
        if action not in POSITIVE_FEEDBACK | NEGATIVE_FEEDBACK:
            continue
        seen_feedback.add(file_id)
        if action in POSITIVE_FEEDBACK:
            signal = _decay(item.created_at, now, 60.0)
            file_feedback[file_id] = signal
        elif action in NEGATIVE_FEEDBACK:
            signal = -_decay(item.created_at, now, 60.0)
            file_feedback[file_id] = signal
            if action == "dismissed" and _decay(item.created_at, now, 30.0) >= 0.5:
                dismissed.append(file_id)
        if extension and action in POSITIVE_FEEDBACK | NEGATIVE_FEEDBACK:
            extension_feedback_weights[extension.lower()] += signal
        if root_id and action in POSITIVE_FEEDBACK | NEGATIVE_FEEDBACK:
            root_feedback_weights[str(root_id)] += signal

    explicit_extensions = tuple(
        sorted(
            {
                str(item).lower() if str(item).startswith(".") else f".{item}".lower()
                for item in preferences.get("preferred_extensions", [])
            }
        )
    )
    top_extensions = tuple(
        value
        for value, _ in sorted(extension_weights.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
    )
    roots = tuple(
        value for value, _ in sorted(root_weights.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    )
    has_history = bool(recent_files or file_feedback)
    return UserProfile(
        preferences=preferences,
        preferred_extensions=explicit_extensions or top_extensions,
        frequent_root_ids=roots,
        recent_file_ids=tuple(recent_files[:50]),
        file_access_weights={key: min(value, 1.0) for key, value in access_weights.items()},
        file_access_counts=dict(access_counts),
        file_feedback=file_feedback,
        extension_feedback={
            key: min(1.0, max(-1.0, value)) for key, value in extension_feedback_weights.items()
        },
        root_feedback={
            key: min(1.0, max(-1.0, value)) for key, value in root_feedback_weights.items()
        },
        dismissed_file_ids=tuple(dismissed),
        profile_status="available" if has_history else "cold_start",
    )


def empty_profile(*, status: str = "cold_start") -> UserProfile:
    """Return safe defaults when history is empty or a profile read fails."""
    return UserProfile(preferences=dict(DEFAULT_PREFERENCES), profile_status=status)


def cached_user_profile(
    session: Session, user_id: uuid.UUID, *, now: datetime | None = None
) -> UserProfile:
    """Reuse the local profile briefly; writes invalidate it and TTL bounds cross-worker staleness."""
    current_time = time.monotonic()
    with _PROFILE_CACHE_LOCK:
        cached = _PROFILE_CACHE.get(user_id)
        if cached is not None and cached[0] > current_time:
            return cached[1]
    profile = build_user_profile(session, user_id, now=now)
    with _PROFILE_CACHE_LOCK:
        if len(_PROFILE_CACHE) >= 512:
            _PROFILE_CACHE.clear()
        _PROFILE_CACHE[user_id] = (current_time + _PROFILE_CACHE_TTL_SECONDS, profile)
    return profile


def invalidate_profile_cache(user_id: uuid.UUID) -> None:
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.pop(user_id, None)
