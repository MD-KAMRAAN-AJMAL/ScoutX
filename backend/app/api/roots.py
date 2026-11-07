"""Authorized indexing root API contracts and route skeletons."""

from fastapi import APIRouter, Path
from pydantic import Field

from app.api.common import APIModel, ErrorResponse, not_implemented

router = APIRouter(prefix="/roots", tags=["roots"])


class RootCreateRequest(APIModel):
    path: str = Field(min_length=1, max_length=4096)
    display_name: str | None = Field(default=None, max_length=200)


class RootRecord(APIModel):
    id: str
    path: str
    display_name: str | None = None
    status: str


class RootListResponse(APIModel):
    items: list[RootRecord]


@router.get("", response_model=RootListResponse, responses={501: {"model": ErrorResponse}})
async def list_roots() -> RootListResponse:
    not_implemented("Authorized root listing")


@router.post(
    "",
    response_model=RootRecord,
    status_code=201,
    responses={501: {"model": ErrorResponse}},
)
async def register_root(_: RootCreateRequest) -> RootRecord:
    not_implemented("Authorized root registration")


@router.delete(
    "/{root_id}",
    status_code=204,
    responses={501: {"model": ErrorResponse}},
)
async def delete_root(root_id: str = Path(min_length=1, max_length=100)) -> None:
    not_implemented(f"Root removal for {root_id}")
