"""Recommendation feedback API contracts and route skeleton."""

from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from app.api.common import APIModel, ErrorResponse, not_implemented

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(APIModel):
    file_id: str = Field(min_length=1, max_length=100)
    action: Literal["opened", "helpful", "not_helpful", "dismissed"]
    query: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(APIModel):
    accepted: bool


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=202,
    responses={501: {"model": ErrorResponse}},
)
async def record_feedback(_: FeedbackRequest) -> FeedbackResponse:
    not_implemented("Recommendation feedback")
