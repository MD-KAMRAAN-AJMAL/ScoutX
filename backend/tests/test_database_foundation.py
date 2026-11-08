from sqlalchemy import Text
from sqlalchemy.orm import configure_mappers

from app.database.models import Base, DocumentChunk, File, IndexRoot
from app.database.schemas import DocumentChunkCreate, RecommendationFeedbackCreate


def test_all_initial_relational_models_are_registered() -> None:
    expected = {
        "users",
        "index_roots",
        "files",
        "document_chunks",
        "index_jobs",
        "file_activity",
        "user_preferences",
        "projects",
        "project_files",
        "agent_runs",
        "recommendation_feedback",
    }
    assert set(Base.metadata.tables) == expected


def test_file_chunk_schema_and_fuzzy_filename_index() -> None:
    configure_mappers()
    assert File.__table__.c.id.primary_key
    assert {
        "user_id",
        "root_id",
        "canonical_path",
        "device_id",
        "inode",
        "content_hash",
        "version",
    } <= set(File.__table__.c.keys())
    assert isinstance(DocumentChunk.__table__.c.content.type, Text)
    assert any(
        index.name == "ix_files_filename_trgm"
        and index.dialect_options["postgresql"].get("using") == "gin"
        for index in File.__table__.indexes
    )
    assert any(constraint.name == "fk_files_root_user" for constraint in File.__table__.constraints)
    assert IndexRoot.__mapper__.relationships["files"]


def test_transfer_schemas_validate_ranges() -> None:
    valid_chunk = DocumentChunkCreate(chunk_index=0, content="source", line_start=1, line_end=3)
    assert valid_chunk.chunk_index == 0
    try:
        RecommendationFeedbackCreate(file_id="00000000-0000-0000-0000-000000000001", rating=6)
    except ValueError:
        pass
    else:
        raise AssertionError("feedback ratings outside 1..5 must be rejected")
