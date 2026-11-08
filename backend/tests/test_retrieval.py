from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient, models

from app.retrieval.code import search_code
from app.retrieval.common import point_candidates, reciprocal_rank_fusion
from app.retrieval.filename import search_exact_filename, search_fuzzy_filename
from app.retrieval.hybrid import search_hybrid
from app.retrieval.keyword import search_keyword
from app.retrieval.media import search_media
from app.retrieval.metadata import search_metadata
from app.retrieval.semantic import search_semantic
from app.vectorstore.qdrant_client import QdrantStore


@pytest.fixture(autouse=True)
def readable_fixture_files(monkeypatch):
    from app.retrieval import common

    monkeypatch.setattr(common, "require_readable_file", lambda path: path)


USER = UUID("00000000-0000-0000-0000-000000000001")


def record(
    filename: str,
    extension: str,
    content: str,
    *,
    modality: str = "text",
    language: str | None = None,
    start_time_ms: int | None = None,
):
    file_id, chunk_id, point_id = uuid4(), uuid4(), uuid4()
    file = SimpleNamespace(
        id=file_id,
        user_id=USER,
        filename=filename,
        canonical_path=f"/authorized/{filename}",
        extension=extension,
        version=1,
        indexing_status="indexed",
        modified_at=None,
        root_id=uuid4(),
    )
    chunk = SimpleNamespace(
        id=chunk_id,
        file_id=file_id,
        qdrant_point_id=point_id,
        content_version=1,
        content=content,
        modality=modality,
        language=language,
        page_number=1 if extension == ".pdf" else None,
        line_start=10 if modality == "code" else None,
        line_end=20 if modality == "code" else None,
        start_time_ms=start_time_ms,
        end_time_ms=(start_time_ms + 5000) if start_time_ms is not None else None,
        chunk_metadata={"content_type": "code_declaration"},
    )
    return file, chunk


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(all=lambda: self.rows)


class FakeEmbeddings:
    def embed_queries(self, texts):
        assert texts == ["matrix multiplication"] or len(texts) == 1
        return [[0.1, 0.2, 0.3]]


class CollectionEmbeddings:
    def embed_queries(self, texts):
        assert len(texts) == 1
        return [[1.0] + [0.0] * 1023]


class FakeStore:
    collection_name = "test_collection"

    def __init__(self, points):
        self.points = points
        self.calls = []
        self.client = self

    def ensure_ready(self):
        pass

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)


def point(point_id, score=0.7):
    return SimpleNamespace(
        id=str(point_id), score=score, payload={"file_id": "ignored; PostgreSQL is authoritative"}
    )


def test_standard_candidate_resolution_preserves_source_references():
    file, chunk = record(
        "lecture.mp4", ".mp4", "matrix multiplication", modality="audio", start_time_ms=42_000
    )
    session = FakeSession([(chunk, file)])
    result = point_candidates(
        session, USER, [point(chunk.qdrant_point_id)], retrieval_method="semantic"
    )
    assert result[0].chunk_id == chunk.id
    assert result[0].file_id == file.id
    assert result[0].start_time_ms == 42_000
    assert result[0].end_time_ms == 47_000
    assert result[0].retrieval_method == "semantic"


def test_candidate_contains_postgres_source_references():
    file, chunk = record("paper.pdf", ".pdf", "Finding")
    chunk.chunk_metadata["source_references"] = [{"page_number": 1, "section_title": "Finding"}]
    session = FakeSession([(chunk, file)])

    result = point_candidates(
        session, USER, [point(chunk.qdrant_point_id)], retrieval_method="semantic"
    )

    assert result[0].source_references == [{"page_number": 1, "section_title": "Finding"}]


def test_exact_and_fuzzy_filename_search_return_candidates():
    file, chunk = record("main.py", ".py", "def main(): pass", modality="code", language="python")
    session = FakeSession([(chunk, file)])
    exact = search_exact_filename(session, USER, "MAIN.PY")
    assert exact[0].file_name == "main.py"
    assert exact[0].retrieval_method == "filename_exact"

    fuzzy_session = FakeSession([(chunk, file, 0.76)])
    fuzzy = search_fuzzy_filename(fuzzy_session, USER, "mian.py")
    assert fuzzy[0].relevance_score == 0.76
    assert fuzzy[0].retrieval_method == "filename_fuzzy"


def test_metadata_search_filters_extension_and_preserves_pdf_page():
    file, chunk = record("notes.pdf", ".pdf", "IO-aware attention", modality="text")
    session = FakeSession([(chunk, file)])
    results = search_metadata(session, USER, extensions=["pdf"])
    assert results[0].page_number == 1
    assert results[0].retrieval_method == "metadata"
    assert any(
        ".pdf" in value
        for value in session.statements[0].compile().params.values()
        if isinstance(value, (list, tuple))
    )


def test_keyword_search_uses_user_scoped_qdrant_text_index():
    file, chunk = record(
        "solver.cpp", ".cpp", "#pragma omp parallel for", modality="code", language="cpp"
    )
    session, store = FakeSession([(chunk, file)]), FakeStore([point(chunk.qdrant_point_id)])
    results = search_keyword(
        session, USER, "omp parallel for", extensions=[".cpp"], modalities=["code"], store=store
    )
    assert results[0].retrieval_method == "keyword"
    assert store.calls[0]["using"] == "content"
    assert store.calls[0]["query_filter"].must[0].match.value == str(USER)
    assert store.calls[0]["query_filter"].must[1].key == "is_active"


def test_dense_semantic_search_uses_existing_embedding_and_vector_interface():
    file, chunk = record("paper.pdf", ".pdf", "FlashAttention IO-awareness")
    session, store = FakeSession([(chunk, file)]), FakeStore([point(chunk.qdrant_point_id)])
    results = search_semantic(
        session,
        USER,
        "FlashAttention IO-awareness",
        embedding_service=FakeEmbeddings(),
        store=store,
    )
    assert results[0].retrieval_method == "semantic"
    assert store.calls[0]["using"] == "dense"
    assert store.calls[0]["query"] == [0.1, 0.2, 0.3]


def test_hybrid_uses_qdrant_rrf_and_never_adds_raw_scores():
    file, chunk = record("paper.pdf", ".pdf", "FlashAttention IO-awareness")
    session, store = FakeSession([(chunk, file)]), FakeStore([point(chunk.qdrant_point_id, 0.03)])
    results = search_hybrid(
        session,
        USER,
        "FlashAttention IO-awareness",
        embedding_service=FakeEmbeddings(),
        store=store,
    )
    query = store.calls[0]
    assert len(query["prefetch"]) == 2
    assert query["query"].fusion.value == "rrf"
    assert results[0].relevance_score == 0.03
    assert results[0].retrieval_method == "hybrid_rrf"
    fused = reciprocal_rank_fusion(results, results)
    assert fused[0].relevance_score < 1


def test_code_search_filters_source_files_and_fuses_tree_sitter_chunks():
    file, chunk = record(
        "solver.cpp", ".cpp", "int max_flow() { return 1; }", modality="code", language="cpp"
    )
    session, store = FakeSession([(chunk, file)]), FakeStore([point(chunk.qdrant_point_id)])
    results = search_code(
        session,
        USER,
        "max_flow",
        languages=["C++"],
        embedding_service=FakeEmbeddings(),
        store=store,
    )
    assert results[0].line_start == 10
    assert results[0].retrieval_method == "code_rrf"
    assert any(
        ".cpp" in value
        for value in session.statements[0].compile().params.values()
        if isinstance(value, (list, tuple))
    )


def test_media_searches_ocr_and_transcripts_with_timestamp_provenance():
    file, chunk = record(
        "lecture.mp4", ".mp4", "matrix multiplication", modality="audio", start_time_ms=120_000
    )
    session, store = FakeSession([(chunk, file)]), FakeStore([point(chunk.qdrant_point_id)])
    results = search_media(
        session,
        USER,
        "matrix multiplication",
        media_types=["video"],
        mode="hybrid",
        embedding_service=FakeEmbeddings(),
        store=store,
    )
    assert results[0].start_time_ms == 120_000
    assert results[0].retrieval_method == "media_rrf"
    assert "image" not in [
        condition.match.value
        for condition in store.calls[0]["query_filter"].must
        if hasattr(condition.match, "value")
    ]


def test_media_searches_image_ocr_chunks():
    file, chunk = record("diagram.png", ".png", "matrix multiplication dataflow", modality="image")
    session, store = FakeSession([(chunk, file)]), FakeStore([point(chunk.qdrant_point_id)])
    results = search_media(
        session,
        USER,
        "matrix multiplication",
        media_types=["image"],
        mode="keyword",
        store=store,
    )
    assert results[0].file_name == "diagram.png"
    assert results[0].modality == "image"
    assert store.calls[0]["query_filter"].must[2].match.any == ["image"]


def test_search_inputs_reject_invalid_limits_and_invalid_media_modes():
    from pydantic import ValidationError

    from app.retrieval.common import QuerySearchInput

    session = FakeSession([])
    with pytest.raises(ValueError):
        search_exact_filename(session, USER, "main.py", limit=0)
    with pytest.raises(ValueError):
        search_keyword(session, USER, "keyword", limit=1001, store=FakeStore([]))
    with pytest.raises(ValueError):
        search_media(session, USER, "media", mode="unsupported", store=FakeStore([]))
    with pytest.raises(ValidationError):
        QuerySearchInput.model_validate({"query": "text", "user_id": str(USER)})


def test_tool_argument_schema_does_not_expose_authenticated_user_id():
    from app.retrieval.common import QuerySearchInput

    assert "user_id" not in QuerySearchInput.model_json_schema()["properties"]


def test_empty_authorized_file_scope_cannot_remove_qdrant_scope():
    from app.retrieval.common import qdrant_filter

    query_filter = qdrant_filter(USER, file_ids=[])
    conditions = {condition.key: condition for condition in query_filter.must}

    assert conditions["user_id"].match.value == str(USER)
    assert conditions["is_active"].match.value is True
    assert conditions["file_id"].match.any == [str(UUID(int=0))]


def test_unreadable_candidates_are_removed_from_every_strategy(monkeypatch):
    from app.retrieval import common

    monkeypatch.setattr(
        common,
        "require_readable_file",
        lambda _: (_ for _ in ()).throw(PermissionError("denied")),
    )
    file, chunk = record("secret.pdf", ".pdf", "private content")
    session = FakeSession([(chunk, file)])

    assert (
        point_candidates(session, USER, [point(chunk.qdrant_point_id)], retrieval_method="keyword")
        == []
    )


def test_external_search_failures_use_stable_backend_error():
    from app.retrieval.common import RetrievalBackendError

    class UnavailableStore(FakeStore):
        def query_points(self, **kwargs):
            raise RuntimeError("private backend detail")

    with pytest.raises(RetrievalBackendError, match="search_keyword backend failed"):
        search_keyword(FakeSession([]), USER, "query", store=UnavailableStore([]))


def test_native_retrieval_on_multimodal_local_qdrant_collection():
    """Exercise text, dense and server-side RRF against all target content types."""
    examples = [
        record("attention.pdf", ".pdf", "matrix multiplication and flash attention"),
        record(
            "solver.cpp",
            ".cpp",
            "int matrix_multiply() { return 1; }",
            modality="code",
            language="cpp",
        ),
        record(
            "solver.py",
            ".py",
            "def matrix_multiply(): return 1",
            modality="code",
            language="python",
        ),
        record("diagram.png", ".png", "matrix multiplication dataflow", modality="image"),
        record(
            "lecture.mp4",
            ".mp4",
            "matrix multiplication discussion",
            modality="audio",
            start_time_ms=75_000,
        ),
    ]
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, collection_name=f"retrieval_test_{uuid4().hex[:10]}")
    try:
        store.ensure_ready()
        client.upsert(
            collection_name=store.collection_name,
            points=[
                models.PointStruct(
                    id=str(chunk.qdrant_point_id),
                    vector={"dense": [1.0] + [0.0] * 1023},
                    payload={
                        "file_id": str(file.id),
                        "user_id": str(USER),
                        "filename": file.filename,
                        "canonical_path": file.canonical_path,
                        "extension": file.extension,
                        "content": chunk.content,
                        "modality": chunk.modality,
                        "language": chunk.language,
                        "content_version": 1,
                        "is_active": True,
                    },
                )
                for file, chunk in examples
            ],
            wait=True,
        )
        session = FakeSession([(chunk, file) for file, chunk in examples])

        keyword = search_keyword(session, USER, "matrix multiplication", store=store)
        dense = search_semantic(
            session,
            USER,
            "matrix multiplication",
            embedding_service=CollectionEmbeddings(),
            store=store,
        )
        hybrid = search_hybrid(
            session,
            USER,
            "matrix multiplication",
            embedding_service=CollectionEmbeddings(),
            store=store,
        )

        assert {candidate.extension for candidate in keyword} == {
            ".pdf",
            ".cpp",
            ".py",
            ".png",
            ".mp4",
        }
        assert len(dense) == 5
        assert len(hybrid) == 5
        assert next(item for item in hybrid if item.extension == ".mp4").start_time_ms == 75_000
    finally:
        client.close()
