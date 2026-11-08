from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from qdrant_client import models
from sqlalchemy.dialects import postgresql

from app.database.repositories import DocumentChunkRepository, FileRepository
from app.embeddings.qwen_embedding import EMBEDDING_DIMENSIONS
from app.vectorstore.qdrant_client import QdrantStore


class FakeScalars:
    def __init__(self, values: list[Any] | None = None) -> None:
        self.values = values or []

    def all(self) -> list[Any]:
        return self.values


class FakeRows:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def all(self) -> list[Any]:
        return self.values


class QuerySession:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.rows = rows or []
        self.statement = None

    def scalars(self, statement: Any) -> FakeScalars:
        self.statement = statement
        return FakeScalars()

    def execute(self, statement: Any) -> FakeRows:
        self.statement = statement
        return FakeRows(self.rows)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None
        self.points: list[Any] = []

    def query_points(self, **kwargs: Any) -> Any:
        self.request = kwargs
        return SimpleNamespace(points=[SimpleNamespace(id=str(uuid.uuid4()), score=0.91)])

    def upsert(self, **kwargs: Any) -> None:
        self.points = kwargs["points"]

    def retrieve(self, **kwargs: Any) -> list[Any]:
        return [
            SimpleNamespace(id=point.id, payload=point.payload)
            for point in self.points
            if point.id in kwargs["ids"]
        ]


def _conditions_by_key(query_filter: models.Filter) -> dict[str, Any]:
    return {item.key: item for item in query_filter.must if hasattr(item, "key")}


def test_dense_search_applies_user_active_project_and_metadata_filters() -> None:
    client = FakeQdrantClient()
    store = QdrantStore(client=client)
    store.ensure_ready = lambda: None  # type: ignore[method-assign]
    user_id, project_id, file_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    hits = store.dense_search(
        [0.0] * EMBEDDING_DIMENSIONS,
        user_id=user_id,
        project_id=project_id,
        authorized_file_ids=[file_id],
        extensions=[".pdf"],
        mime_types=["application/pdf"],
    )

    assert len(hits) == 1
    assert client.request is not None
    conditions = _conditions_by_key(client.request["query_filter"])
    assert conditions["user_id"].match.value == str(user_id)
    assert conditions["is_active"].match.value is True
    assert conditions["file_id"].match.any == [str(file_id)]
    assert conditions["extension"].match.any == [".pdf"]
    assert conditions["mime_type"].match.any == ["application/pdf"]


def test_project_vector_search_requires_postgres_authorized_file_scope() -> None:
    store = QdrantStore(client=FakeQdrantClient())
    with pytest.raises(ValueError, match="PostgreSQL-authorized"):
        store.dense_search(
            [0.0] * EMBEDDING_DIMENSIONS,
            user_id=uuid.uuid4(),
            project_id=uuid.uuid4(),
        )


def test_bm25_search_uses_qdrant_sparse_bm25_vector() -> None:
    client = FakeQdrantClient()
    store = QdrantStore(client=client)
    store.ensure_ready = lambda: None  # type: ignore[method-assign]
    user_id = uuid.uuid4()

    hits = store.bm25_search("project planning", user_id=user_id)

    assert len(hits) == 1
    assert client.request is not None
    assert client.request["using"] == "bm25"
    assert client.request["query"].text == "project planning"
    assert client.request["query"].model == "Qdrant/bm25"
    conditions = _conditions_by_key(client.request["query_filter"])
    assert conditions["user_id"].match.value == str(user_id)
    assert conditions["is_active"].match.value is True


def test_upsert_indexes_content_for_bm25_and_dense_search() -> None:
    client = FakeQdrantClient()
    store = QdrantStore(client=client)
    store.ensure_ready = lambda: None  # type: ignore[method-assign]
    chunk_id, user_id = uuid.uuid4(), uuid.uuid4()
    chunk = {
        "id": str(chunk_id),
        "payload": {
            "content": "database adapter reference",
            "embedding_fingerprint": "abc",
            "user_id": str(user_id),
        },
    }

    result = store.upsert_chunks(user_id, [chunk], [[0.0] * EMBEDDING_DIMENSIONS])

    assert result == [chunk_id]
    assert len(client.points) == 1
    assert client.points[0].vector["dense"] == [0.0] * EMBEDDING_DIMENSIONS
    assert client.points[0].vector["bm25"].text == "database adapter reference"
    assert client.points[0].vector["bm25"].model == "Qdrant/bm25"
    assert client.points[0].payload["bm25_index_version"] == 1


def test_legacy_dense_only_point_is_marked_for_reindex() -> None:
    client = FakeQdrantClient()
    store = QdrantStore(client=client)
    store.ensure_ready = lambda: None  # type: ignore[method-assign]
    user_id, point_id = uuid.uuid4(), uuid.uuid4()
    client.points = [
        SimpleNamespace(
            id=str(point_id),
            payload={"user_id": str(user_id), "embedding_fingerprint": "same-content"},
        )
    ]

    assert store.existing_fingerprints(user_id, [point_id]) == {}


def test_repository_exact_and_fuzzy_file_searches_are_user_scoped() -> None:
    session = QuerySession()
    repository = FileRepository(session)  # type: ignore[arg-type]
    user_id = uuid.uuid4()

    repository.find_exact_filename(user_id, "Notes.PDF")
    exact_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "files.user_id" in exact_sql

    repository.find_fuzzy_filename(user_id, "notes.pdf")
    fuzzy_sql = str(session.statement.compile(dialect=postgresql.dialect()))
    assert "files.user_id" in fuzzy_sql
    assert "similarity(files.filename" in fuzzy_sql


def test_metadata_filter_uses_user_project_type_and_date_scope() -> None:
    session = QuerySession()
    repository = FileRepository(session)  # type: ignore[arg-type]
    user_id, project_id = uuid.uuid4(), uuid.uuid4()

    repository.filter_metadata(
        user_id,
        extensions=[".PDF"],
        mime_types=["application/pdf"],
        modified_after=datetime(2026, 1, 1, tzinfo=UTC),
        modified_before=datetime(2026, 12, 31, tzinfo=UTC),
        project_id=project_id,
    )
    sql = str(session.statement.compile(dialect=postgresql.dialect()))

    assert "files.user_id" in sql
    assert "files.extension IN" in sql
    assert "files.mime_type IN" in sql
    assert "files.modified_at >=" in sql
    assert "files.modified_at <=" in sql
    assert "projects.user_id" in sql
    assert "projects.id =" in sql


def test_chunk_hits_hydrate_postgres_source_and_keep_user_scope() -> None:
    user_id, file_id, point_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    chunk = SimpleNamespace(
        id=uuid.uuid4(),
        file_id=file_id,
        chunk_index=3,
        content="source body",
        content_version=4,
        chunk_metadata={"source_references": [{"page_number": 7, "section_title": "Methods"}]},
    )
    file = SimpleNamespace(
        id=file_id,
        user_id=user_id,
        root_id=uuid.uuid4(),
        canonical_path="/home/user/report.pdf",
        filename="report.pdf",
        version=4,
        indexing_status="indexed",
    )
    session = QuerySession(rows=[(chunk, file)])
    repository = DocumentChunkRepository(session)  # type: ignore[arg-type]

    records = repository.get_active_by_point_ids(user_id, [point_id])
    sql = str(session.statement.compile(dialect=postgresql.dialect()))

    assert "files.user_id" in sql
    assert "document_chunks.content_version = files.version" in sql
    assert records[0].chunk is chunk
    assert records[0].file is file
    assert records[0].file.canonical_path == "/home/user/report.pdf"
    assert records[0].source_references == [{"page_number": 7, "section_title": "Methods"}]
