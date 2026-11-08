"""User-controlled personalization preferences, profile, and activity history."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field, field_validator
from sqlalchemy.orm import Session

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.database.repositories import (
    ActivityRepository,
    IndexRootRepository,
    PreferenceRepository,
    RecommendationFeedbackRepository,
)
from app.personalization.profile import DEFAULT_PREFERENCES, cached_user_profile

router = APIRouter(tags=["personalization"])


class PreferencesResponse(APIModel):
    personalization_enabled: bool = True
    preferred_extensions: list[str] = Field(default_factory=list)
    excluded_extensions: list[str] = Field(default_factory=list)
    preferred_root_ids: list[uuid.UUID] = Field(default_factory=list)
    excluded_root_ids: list[uuid.UUID] = Field(default_factory=list)
    recommendation_limit: int = Field(default=12, ge=1, le=50)


class PreferencesUpdate(PreferencesResponse):
    @field_validator("preferred_extensions", "excluded_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            item = value.strip().lower()
            if not item:
                raise ValueError("extensions cannot be blank")
            item = item if item.startswith(".") else f".{item}"
            if len(item) > 32:
                raise ValueError("extensions must be at most 32 characters")
            normalized.append(item)
        return sorted(set(normalized))

    @field_validator("preferred_root_ids", "excluded_root_ids")
    @classmethod
    def unique_roots(cls, values: list[uuid.UUID]) -> list[uuid.UUID]:
        return list(dict.fromkeys(values))


class PersonalizationProfileResponse(APIModel):
    status: Literal["available", "cold_start", "disabled"]
    personalization_enabled: bool
    preferred_extensions: list[str]
    frequent_roots: list[dict[str, str]]
    recent_file_count: int = Field(ge=0)
    feedback_signal_count: int = Field(ge=0)


class ActivityItem(APIModel):
    id: uuid.UUID
    event_type: str
    file_id: uuid.UUID | None = None
    filename: str | None = None
    extension: str | None = None
    created_at: datetime


class ActivityListResponse(APIModel):
    items: list[ActivityItem]
    limit: int
    offset: int
    total: int


class ClearActivityResponse(APIModel):
    deleted_activity_events: int
    deleted_feedback_items: int


class ResetRequest(APIModel):
    clear_activity_history: bool = False


class ResetResponse(APIModel):
    reset: bool
    activity_history_cleared: bool


ActivityEvent = Literal[
    "search_performed",
    "search_result_opened",
    "file_opened",
    "file_previewed",
    "opened",
    "selected",
    "recommendation_clicked",
    "recommendation_dismissed",
    "feedback_positive",
    "feedback_negative",
]


def _preference_record(session: Session, user_id: uuid.UUID) -> dict:
    record = PreferenceRepository(session).get(user_id)
    return (
        {**DEFAULT_PREFERENCES, **(record.preferences or {})}
        if record
        else dict(DEFAULT_PREFERENCES)
    )


def _public_preferences(values: dict) -> PreferencesResponse:
    allowed = PreferencesResponse.model_fields
    return PreferencesResponse(**{key: value for key, value in values.items() if key in allowed})


@router.get(
    "/personalization/preferences",
    response_model=PreferencesResponse,
    responses={401: {"model": ErrorResponse}},
)
def get_preferences(
    session: Session = Depends(get_db), user_id: uuid.UUID = Depends(get_local_user_id)
) -> PreferencesResponse:
    return _public_preferences(_preference_record(session, user_id))


@router.put(
    "/personalization/preferences",
    response_model=PreferencesResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def update_preferences(
    request: PreferencesUpdate,
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> PreferencesResponse:
    for root_id in set(request.preferred_root_ids + request.excluded_root_ids):
        root = IndexRootRepository(session).get(user_id, root_id)
        if root is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_preference_root",
                    "message": "A selected folder is not available in this workspace.",
                },
            )
    previous = _preference_record(session, user_id)
    updated = request.model_dump(mode="json")
    if "profile_reset_at" in previous:
        updated["profile_reset_at"] = previous["profile_reset_at"]
    PreferenceRepository(session).upsert(user_id, updated)
    session.commit()
    return _public_preferences(updated)


@router.get(
    "/personalization/profile",
    response_model=PersonalizationProfileResponse,
    responses={401: {"model": ErrorResponse}},
)
def get_profile(
    session: Session = Depends(get_db), user_id: uuid.UUID = Depends(get_local_user_id)
) -> PersonalizationProfileResponse:
    profile = cached_user_profile(session, user_id)
    if not profile.enabled:
        status = "disabled"
    elif not profile.recent_file_ids and not profile.file_feedback:
        status = "cold_start"
    else:
        status = "available"
    roots = []
    for root_id in profile.frequent_root_ids:
        try:
            root = IndexRootRepository(session).get(user_id, uuid.UUID(root_id))
        except ValueError:
            root = None
        if root is not None:
            roots.append({"id": str(root.id), "name": root.display_name or "Authorized folder"})
    return PersonalizationProfileResponse(
        status=status,
        personalization_enabled=profile.enabled,
        preferred_extensions=list(profile.preferred_extensions),
        frequent_roots=roots,
        recent_file_count=len(profile.recent_file_ids),
        feedback_signal_count=len(profile.file_feedback),
    )


@router.get(
    "/activity",
    response_model=ActivityListResponse,
    responses={401: {"model": ErrorResponse}},
)
def list_activity(
    event_type: ActivityEvent | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> ActivityListResponse:
    repository = ActivityRepository(session)
    rows = repository.list_for_user(user_id, limit=limit, offset=offset, event_type=event_type)
    items = [
        ActivityItem(
            id=event.id,
            event_type=event.event_type,
            file_id=event.file_id,
            filename=filename,
            extension=extension,
            created_at=event.created_at,
        )
        for event, filename, extension in rows
    ]
    return ActivityListResponse(
        items=items,
        limit=limit,
        offset=offset,
        total=repository.count_for_user(user_id, event_type=event_type),
    )


@router.delete(
    "/activity",
    response_model=ClearActivityResponse,
    responses={401: {"model": ErrorResponse}},
)
def clear_activity(
    session: Session = Depends(get_db), user_id: uuid.UUID = Depends(get_local_user_id)
) -> ClearActivityResponse:
    deleted_events = ActivityRepository(session).clear_for_user(user_id)
    deleted_feedback = RecommendationFeedbackRepository(session).clear_for_user(user_id)
    session.commit()
    return ClearActivityResponse(
        deleted_activity_events=deleted_events, deleted_feedback_items=deleted_feedback
    )


@router.post(
    "/personalization/reset",
    response_model=ResetResponse,
    responses={401: {"model": ErrorResponse}},
)
def reset_personalization(
    request: ResetRequest,
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> ResetResponse:
    preferences = _preference_record(session, user_id)
    preferences["profile_reset_at"] = datetime.now(UTC).isoformat()
    if request.clear_activity_history:
        ActivityRepository(session).clear_for_user(user_id)
        RecommendationFeedbackRepository(session).clear_for_user(user_id)
    PreferenceRepository(session).upsert(user_id, preferences)
    session.commit()
    return ResetResponse(reset=True, activity_history_cleared=request.clear_activity_history)
