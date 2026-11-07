"""Shared, provenance-preserving output contract for every content parser."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractionStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class ExtractedBlock(BaseModel):
    """One searchable unit with source location and parser provenance."""

    model_config = ConfigDict(extra="allow")

    content: str
    content_type: str = "text"
    file_id: UUID
    content_version: int = Field(ge=1)
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)
    sheet_name: str | None = None
    cell_range: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    start_time_ms: int | None = Field(default=None, ge=0)
    end_time_ms: int | None = Field(default=None, ge=0)
    language: str | None = None
    section_title: str | None = None
    parser_name: str
    parser_version: str
    extraction_warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ranges(self) -> ExtractedBlock:
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        if self.start_time_ms is not None and self.end_time_ms is not None and self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must not precede start_time_ms")
        return self


class ExtractedDocument(BaseModel):
    """Consistent result returned by all format parsers."""

    model_config = ConfigDict(extra="forbid")

    file_id: UUID
    content_version: int = Field(ge=1)
    status: ExtractionStatus
    blocks: list[ExtractedBlock] = Field(default_factory=list)
    parser_name: str
    parser_version: str
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


def make_block(
    content: str,
    *,
    file_id: UUID,
    content_version: int,
    parser_name: str,
    parser_version: str,
    **kwargs: Any,
) -> ExtractedBlock:
    """Build a block while ensuring mandatory extraction provenance is present."""
    return ExtractedBlock(
        content=content,
        file_id=file_id,
        content_version=content_version,
        parser_name=parser_name,
        parser_version=parser_version,
        **kwargs,
    )
