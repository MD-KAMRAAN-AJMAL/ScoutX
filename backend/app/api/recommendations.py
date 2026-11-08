"""User-scoped personalized file recommendations."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field
from sqlalchemy.orm import Session

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.personalization.recommendations import RecommendationCategory, RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationItem(APIModel):
    id: uuid.UUID
    file_id: uuid.UUID
    filename: str
    path: str
    extension: str | None = None
    mime_type: str | None = None
    modified_at: datetime | None = None
    score: float = Field(ge=0, le=1)
    category: RecommendationCategory
    reason: str


class RecommendationListResponse(APIModel):
    items: list[RecommendationItem]
    category: RecommendationCategory
    limit: int


@router.get(
    "",
    response_model=RecommendationListResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_recommendations(
    category: RecommendationCategory = "recommended_for_you",
    limit: int = Query(default=12, ge=1, le=50),
    extension: str | None = Query(default=None, min_length=1, max_length=32),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> RecommendationListResponse:
    try:
        items = RecommendationService(session, user_id).recommend(
            category=category, limit=limit, extension=extension
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_recommendation_request", "message": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "recommendations_unavailable",
                "message": "Recommendations are temporarily unavailable.",
            },
        ) from exc
    return RecommendationListResponse(
        items=[
            RecommendationItem(
                id=item.file_id,
                file_id=item.file_id,
                filename=item.filename,
                path=item.path,
                extension=item.extension,
                mime_type=item.mime_type,
                modified_at=item.modified_at,
                score=item.score,
                category=item.category,
                reason=item.reason,
            )
            for item in items
        ],
        category=category,
        limit=limit,
    )
