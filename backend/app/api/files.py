"""File listing API contracts and route skeleton."""

from fastapi import APIRouter, Query

from app.api.common import APIModel, ErrorResponse, not_implemented

router = APIRouter(prefix="/files", tags=["files"])


class FileRecord(APIModel):
    id: str
    name: str
    path: str
    mime_type: str | None = None
    size_bytes: int | None = None
    modified_at: str | None = None
    indexing_status: str


class FileListResponse(APIModel):
    items: list[FileRecord]
    limit: int
    offset: int
    total: int


@router.get("", response_model=FileListResponse, responses={501: {"model": ErrorResponse}})
async def list_files(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> FileListResponse:
    not_implemented("File listing")
