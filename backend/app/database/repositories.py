"""SQLAlchemy repository interfaces for phase-one persistence operations.

Repositories receive an authenticated application user ID explicitly. They never
accept filesystem paths as authority and always scope user-owned lookups by user ID.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import (
    AgentRun,
    DocumentChunk,
    File,
    FileActivity,
    IndexJob,
    IndexRoot,
    Project,
    ProjectFile,
    RecommendationFeedback,
    User,
    UserPreference,
)
from app.database.schemas import (
    DocumentChunkCreate,
    FileCreate,
    IndexJobCreate,
    IndexRootCreate,
    ProjectCreate,
    RecommendationFeedbackCreate,
    UserCreate,
)


class UserRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, data: UserCreate) -> User:
        user = User(**data.model_dump())
        self.session.add(user)
        self.session.flush()
        return user

    def get(self, user_id: uuid.UUID) -> User | None:
        return self.session.get(User, user_id)


class IndexRootRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: uuid.UUID, data: IndexRootCreate) -> IndexRoot:
        root = IndexRoot(user_id=user_id, **data.model_dump())
        self.session.add(root)
        self.session.flush()
        return root

    def get(self, user_id: uuid.UUID, root_id: uuid.UUID) -> IndexRoot | None:
        return self.session.scalar(select(IndexRoot).where(IndexRoot.id == root_id, IndexRoot.user_id == user_id))

    def list_for_user(self, user_id: uuid.UUID, *, enabled_only: bool = False) -> Sequence[IndexRoot]:
        statement = select(IndexRoot).where(IndexRoot.user_id == user_id).order_by(IndexRoot.created_at)
        if enabled_only:
            statement = statement.where(IndexRoot.is_enabled.is_(True))
        return self.session.scalars(statement).all()


class FileRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: uuid.UUID, data: FileCreate) -> File:
        # Validate root ownership before inserting a file under the caller's identity.
        root = self.session.scalar(
            select(IndexRoot).where(IndexRoot.id == data.root_id, IndexRoot.user_id == user_id)
        )
        if root is None:
            raise LookupError("Index root does not exist for this user")
        file = File(user_id=user_id, **data.model_dump())
        self.session.add(file)
        self.session.flush()
        return file

    def get(self, user_id: uuid.UUID, file_id: uuid.UUID) -> File | None:
        return self.session.scalar(select(File).where(File.id == file_id, File.user_id == user_id))

    def get_by_path(self, user_id: uuid.UUID, root_id: uuid.UUID, canonical_path: str) -> File | None:
        return self.session.scalar(
            select(File).where(
                File.user_id == user_id,
                File.root_id == root_id,
                File.canonical_path == canonical_path,
            )
        )

    def list_for_root(
        self, user_id: uuid.UUID, root_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> Sequence[File]:
        if limit < 1 or limit > 1000 or offset < 0:
            raise ValueError("limit must be 1..1000 and offset must be nonnegative")
        return self.session.scalars(
            select(File)
            .where(File.user_id == user_id, File.root_id == root_id)
            .order_by(File.canonical_path)
            .limit(limit)
            .offset(offset)
        ).all()

    def update_indexing_status(self, user_id: uuid.UUID, file_id: uuid.UUID, status: str) -> File | None:
        file = self.get(user_id, file_id)
        if file is None:
            return None
        if status not in {"pending", "queued", "processing", "indexed", "failed", "excluded", "deleted"}:
            raise ValueError("Unsupported indexing status")
        file.indexing_status = status
        self.session.flush()
        return file


class DocumentChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_version(
        self, user_id: uuid.UUID, file_id: uuid.UUID, chunks: Sequence[DocumentChunkCreate]
    ) -> list[DocumentChunk]:
        file = self.session.scalar(select(File).where(File.id == file_id, File.user_id == user_id))
        if file is None:
            raise LookupError("File does not exist for this user")
        version = file.version
        self.session.query(DocumentChunk).filter(
            DocumentChunk.file_id == file_id, DocumentChunk.content_version == version
        ).delete(synchronize_session=False)
        records = [
            DocumentChunk(
                file_id=file_id,
                **(chunk.model_dump(exclude={"content_version"}) | {"content_version": version}),
            )
            for chunk in chunks
        ]
        self.session.add_all(records)
        self.session.flush()
        return records

    def list_for_file(
        self, user_id: uuid.UUID, file_id: uuid.UUID, *, content_version: int | None = None
    ) -> Sequence[DocumentChunk]:
        statement = select(DocumentChunk).join(File).where(File.user_id == user_id, File.id == file_id)
        if content_version is not None:
            statement = statement.where(DocumentChunk.content_version == content_version)
        return self.session.scalars(statement.order_by(DocumentChunk.chunk_index)).all()


class IndexJobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: uuid.UUID, data: IndexJobCreate) -> IndexJob:
        if data.root_id and self.session.scalar(
            select(IndexRoot.id).where(IndexRoot.id == data.root_id, IndexRoot.user_id == user_id)
        ) is None:
            raise LookupError("Index root does not exist for this user")
        if data.file_id and self.session.scalar(
            select(File.id).where(File.id == data.file_id, File.user_id == user_id)
        ) is None:
            raise LookupError("File does not exist for this user")
        job = IndexJob(user_id=user_id, **data.model_dump())
        self.session.add(job)
        self.session.flush()
        return job

    def list_for_user(self, user_id: uuid.UUID, *, limit: int = 100) -> Sequence[IndexJob]:
        return self.session.scalars(
            select(IndexJob).where(IndexJob.user_id == user_id).order_by(IndexJob.created_at.desc()).limit(limit)
        ).all()


class PreferenceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, user_id: uuid.UUID) -> UserPreference | None:
        return self.session.get(UserPreference, user_id)

    def upsert(self, user_id: uuid.UUID, preferences: dict) -> UserPreference:
        record = self.get(user_id)
        if record is None:
            record = UserPreference(user_id=user_id, preferences=preferences)
            self.session.add(record)
        else:
            record.preferences = preferences
        self.session.flush()
        return record


class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: uuid.UUID, data: ProjectCreate) -> Project:
        project = Project(user_id=user_id, **data.model_dump())
        self.session.add(project)
        self.session.flush()
        return project

    def attach_file(self, user_id: uuid.UUID, project_id: uuid.UUID, file_id: uuid.UUID) -> ProjectFile:
        project = self.session.scalar(select(Project).where(Project.id == project_id, Project.user_id == user_id))
        file = self.session.scalar(select(File).where(File.id == file_id, File.user_id == user_id))
        if project is None or file is None:
            raise LookupError("Project or file does not exist for this user")
        association = ProjectFile(project_id=project_id, file_id=file_id)
        self.session.add(association)
        self.session.flush()
        return association


class ActivityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record(self, user_id: uuid.UUID, event_type: str, *, file_id: uuid.UUID | None = None, details: dict | None = None) -> FileActivity:
        if file_id and self.session.scalar(
            select(File.id).where(File.id == file_id, File.user_id == user_id)
        ) is None:
            raise LookupError("File does not exist for this user")
        activity = FileActivity(user_id=user_id, file_id=file_id, event_type=event_type, details=details or {})
        self.session.add(activity)
        self.session.flush()
        return activity


class AgentRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: uuid.UUID, query: str) -> AgentRun:
        run = AgentRun(user_id=user_id, query=query)
        self.session.add(run)
        self.session.flush()
        return run


class RecommendationFeedbackRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: uuid.UUID, data: RecommendationFeedbackCreate) -> RecommendationFeedback:
        if self.session.scalar(select(File.id).where(File.id == data.file_id, File.user_id == user_id)) is None:
            raise LookupError("File does not exist for this user")
        if data.agent_run_id and self.session.scalar(
            select(AgentRun.id).where(AgentRun.id == data.agent_run_id, AgentRun.user_id == user_id)
        ) is None:
            raise LookupError("Agent run does not exist for this user")
        record = RecommendationFeedback(user_id=user_id, **data.model_dump())
        self.session.add(record)
        self.session.flush()
        return record

