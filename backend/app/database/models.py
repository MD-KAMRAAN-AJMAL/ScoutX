"""Canonical PostgreSQL models for the indexing and recommendation system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    and_,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative metadata used by Alembic and application code."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=text("true"), nullable=False)

    index_roots: Mapped[list[IndexRoot]] = relationship(back_populates="user", cascade="all, delete-orphan")
    files: Mapped[list[File]] = relationship(back_populates="user", cascade="all, delete-orphan")
    preferences: Mapped[UserPreference | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )
    projects: Mapped[list[Project]] = relationship(back_populates="user", cascade="all, delete-orphan")


class IndexRoot(TimestampMixin, Base):
    __tablename__ = "index_roots"
    __table_args__ = (
        UniqueConstraint("user_id", "canonical_path", name="uq_index_roots_user_path"),
        UniqueConstraint("id", "user_id", name="uq_index_roots_id_user"),
        Index("ix_index_roots_user_enabled", "user_id", "is_enabled"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    canonical_path: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    is_enabled: Mapped[bool] = mapped_column(default=True, server_default=text("true"), nullable=False)
    last_reconciled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)

    user: Mapped[User] = relationship(back_populates="index_roots")
    files: Mapped[list[File]] = relationship(
        back_populates="root",
        cascade="all, delete-orphan",
        primaryjoin=lambda: and_(IndexRoot.id == File.root_id, IndexRoot.user_id == File.user_id),
        foreign_keys=lambda: [File.root_id],
    )


class File(TimestampMixin, Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("root_id", "canonical_path", name="uq_files_root_path"),
        ForeignKeyConstraint(
            ["root_id", "user_id"],
            ["index_roots.id", "index_roots.user_id"],
            name="fk_files_root_user",
            ondelete="CASCADE",
        ),
        CheckConstraint("size_bytes >= 0", name="ck_files_size_nonnegative"),
        CheckConstraint("version >= 1", name="ck_files_version_positive"),
        CheckConstraint(
            "indexing_status IN ('pending','queued','processing','indexed','failed','excluded','deleted')",
            name="ck_files_indexing_status",
        ),
        Index("ix_files_user_modified", "user_id", "modified_at"),
        Index("ix_files_root_status", "root_id", "indexing_status"),
        Index("ix_files_content_hash", "user_id", "content_hash"),
        Index("ix_files_filename_trgm", "filename", postgresql_using="gin", postgresql_ops={"filename": "gin_trgm_ops"}),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    root_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    canonical_path: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    extension: Mapped[str | None] = mapped_column(String(32))
    mime_type: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    device_id: Mapped[int | None] = mapped_column(BigInteger)
    inode: Mapped[int | None] = mapped_column(BigInteger)
    content_hash: Mapped[str | None] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    indexing_status: Mapped[str] = mapped_column(String(24), default="pending", server_default="pending", nullable=False)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)

    user: Mapped[User] = relationship(back_populates="files")
    root: Mapped[IndexRoot] = relationship(
        back_populates="files",
        primaryjoin=lambda: and_(IndexRoot.id == File.root_id, IndexRoot.user_id == File.user_id),
        foreign_keys=lambda: [File.root_id],
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(back_populates="file", cascade="all, delete-orphan")
    activities: Mapped[list[FileActivity]] = relationship(back_populates="file")


class DocumentChunk(TimestampMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("file_id", "content_version", "chunk_index", name="uq_chunks_file_version_index"),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_index_nonnegative"),
        CheckConstraint("content_version >= 1", name="ck_chunks_version_positive"),
        CheckConstraint("line_start IS NULL OR line_start >= 1", name="ck_chunks_line_start_positive"),
        CheckConstraint("line_end IS NULL OR line_end >= 1", name="ck_chunks_line_end_positive"),
        Index("ix_chunks_file_version", "file_id", "content_version"),
        Index("ix_chunks_modality_language", "modality", "language"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    modality: Mapped[str] = mapped_column(String(32), default="text", server_default="text", nullable=False)
    language: Mapped[str | None] = mapped_column(String(32))
    page_number: Mapped[int | None] = mapped_column(Integer)
    line_start: Mapped[int | None] = mapped_column(Integer)
    line_end: Mapped[int | None] = mapped_column(Integer)
    start_time_ms: Mapped[int | None] = mapped_column(BigInteger)
    end_time_ms: Mapped[int | None] = mapped_column(BigInteger)
    section_title: Mapped[str | None] = mapped_column(String(512))
    content_version: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    qdrant_point_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), unique=True)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)

    file: Mapped[File] = relationship(back_populates="chunks")


class IndexJob(TimestampMixin, Base):
    __tablename__ = "index_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','succeeded','failed','cancelled')", name="ck_index_jobs_status"),
        CheckConstraint("attempts >= 0", name="ck_index_jobs_attempts_nonnegative"),
        Index("ix_index_jobs_status_created", "status", "created_at"),
        Index("ix_index_jobs_file_status", "file_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    root_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("index_roots.id", ondelete="SET NULL"))
    file_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"))
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued", nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)


class FileActivity(TimestampMixin, Base):
    __tablename__ = "file_activity"
    __table_args__ = (Index("ix_file_activity_user_time", "user_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="SET NULL"))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)

    file: Mapped[File | None] = relationship(back_populates="activities")


class UserPreference(TimestampMixin, Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    preferences: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)
    user: Mapped[User] = relationship(back_populates="preferences")


class Project(TimestampMixin, Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_projects_user_name"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)

    user: Mapped[User] = relationship(back_populates="projects")
    files: Mapped[list[ProjectFile]] = relationship(back_populates="project", cascade="all, delete-orphan")


class ProjectFile(Base):
    __tablename__ = "project_files"

    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(32), default="related", server_default="related", nullable=False)
    project: Mapped[Project] = relationship(back_populates="files")
    file: Mapped[File] = relationship()


class AgentRun(TimestampMixin, Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint("status IN ('running','succeeded','failed','cancelled')", name="ck_agent_runs_status"),
        Index("ix_agent_runs_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="running", server_default="running", nullable=False)
    plan: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)
    outcome: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RecommendationFeedback(TimestampMixin, Base):
    __tablename__ = "recommendation_feedback"
    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_feedback_rating_range"),
        Index("ix_feedback_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="SET NULL"))
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    feedback_text: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str | None] = mapped_column(String(32))

