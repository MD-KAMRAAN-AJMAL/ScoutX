"""User-scoped recommendation feedback persistence."""

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field
from sqlalchemy.orm import Session

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.database.repositories import ActivityRepository, RecommendationFeedbackRepository
from app.database.schemas import RecommendationFeedbackCreate

router = APIRouter(prefix="/feedback", tags=["feedback"])
logger = logging.getLogger(__name__)


class FeedbackRequest(APIModel):
    file_id: uuid.UUID
    action: Literal[
        "opened", "selected", "relevant", "not_relevant", "helpful", "not_helpful", "dismissed"
    ]
    query: str | None = Field(default=None, max_length=2000)
    agent_run_id: uuid.UUID | None = None


class FeedbackResponse(APIModel):
    accepted: bool


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=202,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def record_feedback(
    request: FeedbackRequest,
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> FeedbackResponse:
    rating = (
        5
        if request.action in {"selected", "relevant", "helpful"}
        else 1
        if request.action in {"not_relevant", "not_helpful", "dismissed"}
        else 3
    )
    try:
        RecommendationFeedbackRepository(session).create(
            user_id,
            RecommendationFeedbackCreate(
                file_id=request.file_id,
                agent_run_id=request.agent_run_id,
                rating=rating,
                action=request.action,
                feedback_text=request.query,
            ),
        )
        if request.action in {"opened", "selected"}:
            ActivityRepository(session).record(
                user_id,
                request.action,
                file_id=request.file_id,
                details={"query": request.query} if request.query else {},
            )
        session.commit()
    except LookupError as exc:
        session.rollback()
        raise HTTPException(
            status_code=404,
            detail={
                "code": "feedback_target_not_found",
                "message": "The selected file or search run was not found.",
            },
        ) from exc
    except Exception as exc:
        session.rollback()
        logger.exception("Unable to save feedback for user %s", user_id)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "feedback_unavailable",
                "message": "Feedback could not be saved right now.",
            },
        ) from exc
    return FeedbackResponse(accepted=True)
