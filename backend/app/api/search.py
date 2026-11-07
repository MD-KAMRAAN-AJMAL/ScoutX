"""Search API contracts and route skeleton."""

from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from app.api.common import APIModel, ErrorResponse, not_implemented

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=20, ge=1, le=100)
    mode: Literal["auto", "keyword", "semantic", "hybrid"] = "auto"


class SearchResult(APIModel):
    file_id: str
    name: str
    path: str
    score: float = Field(ge=0, le=1)
    excerpt: str | None = None
    source_reference: str | None = None


class SearchResponse(APIModel):
    query: str
    results: list[SearchResult]
    answer: str | None = None


@router.post(
    "",
    response_model=SearchResponse,
    responses={501: {"model": ErrorResponse}},
)
async def search_files(_: SearchRequest) -> SearchResponse:
    not_implemented("Search")
