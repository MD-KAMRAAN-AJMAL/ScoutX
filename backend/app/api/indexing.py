"""Indexing status API contracts and route skeleton."""

from fastapi import APIRouter

from app.api.common import APIModel, ErrorResponse, not_implemented

router = APIRouter(prefix="/index", tags=["indexing"])


class IndexStatusResponse(APIModel):
    state: str
    files_discovered: int
    files_indexed: int
    files_failed: int
    last_reconciled_at: str | None = None


@router.get(
    "/status",
    response_model=IndexStatusResponse,
    responses={501: {"model": ErrorResponse}},
)
async def indexing_status() -> IndexStatusResponse:
    not_implemented("Indexing status")
