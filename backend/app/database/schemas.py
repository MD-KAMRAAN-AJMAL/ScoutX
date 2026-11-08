"""Pydantic API and repository transfer schemas for the canonical database models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: str | None = None
    display_name: str | None = None


class UserRead(ORMModel):
    id: uuid.UUID
    email: str | None
    display_name: str | None
    is_active: bool
    created_at: datetime


class IndexRootCreate(BaseModel):
    canonical_path: str = Field(min_length=1)
    display_name: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class IndexRootRead(ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    canonical_path: str
    display_name: str | None
    is_enabled: bool
    last_reconciled_at: datetime | None
    created_at: datetime


class FileCreate(BaseModel):
    root_id: uuid.UUID
    canonical_path: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=1024)
    extension: str | None = None
    mime_type: str | None = None
    size_bytes: int = Field(ge=0)
    modified_at: datetime | None = None
    device_id: int | None = None
    inode: int | None = None
    content_hash: str | None = None


class FileRead(ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    root_id: uuid.UUID
    canonical_path: str
    filename: str
    extension: str | None
    mime_type: str | None
    size_bytes: int
    modified_at: datetime | None
    content_hash: str | None
    version: int
    indexing_status: Literal[
        "pending", "queued", "processing", "indexed", "failed", "excluded", "deleted"
    ]
    last_indexed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DocumentChunkCreate(BaseModel):
    chunk_index: int = Field(ge=0)
    content: str
    modality: str = "text"
    language: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    start_time_ms: int | None = Field(default=None, ge=0)
    end_time_ms: int | None = Field(default=None, ge=0)
    section_title: str | None = None
    content_version: int = Field(default=1, ge=1)
    qdrant_point_id: uuid.UUID | None = None
    chunk_metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunkRead(ORMModel):
    id: uuid.UUID
    file_id: uuid.UUID
    chunk_index: int
    content: str
    modality: str
    language: str | None
    page_number: int | None
    line_start: int | None
    line_end: int | None
    start_time_ms: int | None
    end_time_ms: int | None
    section_title: str | None
    content_version: int
    qdrant_point_id: uuid.UUID | None


class IndexJobCreate(BaseModel):
    root_id: uuid.UUID | None = None
    file_id: uuid.UUID | None = None
    job_type: str = Field(min_length=1, max_length=32)
    payload: dict[str, Any] = Field(default_factory=dict)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class RecommendationFeedbackCreate(BaseModel):
    file_id: uuid.UUID
    agent_run_id: uuid.UUID | None = None
    rating: int = Field(ge=1, le=5)
    feedback_text: str | None = None
    action: str | None = None
