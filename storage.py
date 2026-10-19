"""Persistent SQLite storage: schema migrator, UsageLog, UserProfileStore, MetadataIndex, and VectorIndex."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterator, Sequence

from scoutx.embeddings import (
    EmbeddingDimensionMismatchError,
    EmbeddingProvider,
    EmbeddingResponseFormatError,
    build_document_embedding_text,
    validate_and_normalize_vector,
)
from scoutx.models import DateWindow, FileDocument, UserProfile
from scoutx.text import clamp_score

CURRENT_SCHEMA_VERSION = 2

LEGACY_V1_USAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "event_type",
        "document_id",
        "path",
        "extension",
        "topic",
        "raw_query",
        "result_count",
        "created_at",
    }
)


class StorageSchemaError(RuntimeError):
    """Raised when a SQLite database has an unsupported or incompatible schema version."""


@dataclass(frozen=True)
class StoredDocumentMetadata:
    """Catalog entry stored in MetadataIndex.

    NOTE: ``resolved_path`` and ``parent_dir`` are persisted strictly as local
    sensitive metadata for local file resolution and opening. They must never be
    forwarded to remote LLM or embedding providers. Raw document content is
    never stored in MetadataIndex.
    """

    document_id: str
    resolved_path: str
    parent_dir: str
    name: str
    title: str
    extension: str
    is_directory: bool
    size_bytes: int
    modified_at: datetime | None
    created_at: datetime | None
    content_hash: str
    indexed_at: str


def compute_document_content_hash(document: FileDocument) -> str:
    """Compute a deterministic SHA-256 fingerprint of a document's content and state."""
    hasher = hashlib.sha256()
    hasher.update(b"dir\x00" if document.is_directory else b"file\x00")
    hasher.update((document.title or "").encode("utf-8", errors="replace"))
    hasher.update(b"\x00")
    hasher.update((document.extension or "").encode("utf-8", errors="replace"))
    hasher.update(b"\x00")
    content = document.content or ""
    hasher.update(content.encode("utf-8", errors="replace"))
    if not content:
        if document.modified_at is not None:
            hasher.update(b"\x00mtime:")
            hasher.update(document.modified_at.isoformat().encode("utf-8"))
        try:
            stat = document.path.stat()
            hasher.update(f"\x00size:{stat.st_size}".encode("utf-8"))
        except OSError:
            pass
    return hasher.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _immediate_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        connection.execute("COMMIT")


def _create_all_tables_idempotent(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            document_id TEXT,
            path TEXT,
            extension TEXT,
            topic TEXT,
            raw_query TEXT,
            result_count INTEGER,
            lane TEXT,
            forced_exit INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_events_type_document
        ON usage_events (event_type, document_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_events_type_extension
        ON usage_events (event_type, extension)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_file_opens (
            document_id TEXT PRIMARY KEY,
            open_count INTEGER NOT NULL CHECK (open_count >= 0),
            last_opened_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_extension_counts (
            extension TEXT PRIMARY KEY,
            open_count INTEGER NOT NULL CHECK (open_count >= 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_topic_counts (
            topic TEXT PRIMARY KEY,
            topic_count INTEGER NOT NULL CHECK (topic_count >= 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata_documents (
            document_id TEXT PRIMARY KEY,
            resolved_path TEXT NOT NULL,
            parent_dir TEXT NOT NULL,
            name TEXT NOT NULL,
            title TEXT NOT NULL,
            extension TEXT NOT NULL,
            is_directory INTEGER NOT NULL DEFAULT 0,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            modified_at TEXT,
            created_at TEXT,
            content_hash TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_metadata_documents_extension
        ON metadata_documents (extension)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_metadata_documents_modified_at
        ON metadata_documents (modified_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_index_meta (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            model_id TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK (dimensions >= 1),
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_embeddings (
            document_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK (dimensions >= 1),
            vector_blob BLOB NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def ensure_database_schema(
    db_path: str | Path,
    *,
    _migration_fail_hook: Callable[[sqlite3.Connection], None] | None = None,
) -> int:
    """Single authoritative owner of SQLite PRAGMA user_version and migrations for ScoutX."""
    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(resolved)
    try:
        with _immediate_transaction(connection):
            row = connection.execute("PRAGMA user_version").fetchone()
            user_version = int(row[0]) if row is not None else 0

            if user_version < 0 or user_version > CURRENT_SCHEMA_VERSION:
                raise StorageSchemaError(
                    f"Unsupported SQLite schema version {user_version} in '{resolved}'; "
                    f"supported versions are 0..{CURRENT_SCHEMA_VERSION}"
                )

            if user_version == CURRENT_SCHEMA_VERSION:
                _create_all_tables_idempotent(connection)
                return CURRENT_SCHEMA_VERSION

            # user_version is 0 or 1: check whether a legacy usage_events table already exists
            has_usage_events = (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'usage_events'"
                ).fetchone()
                is not None
            )

            if has_usage_events:
                col_rows = connection.execute("PRAGMA table_info(usage_events)").fetchall()
                existing_cols = {str(col[1]) for col in col_rows}
                missing_legacy = LEGACY_V1_USAGE_COLUMNS - existing_cols
                if missing_legacy:
                    raise StorageSchemaError(
                        f"Legacy table 'usage_events' in '{resolved}' is missing required v1 columns: "
                        f"{sorted(missing_legacy)}"
                    )
                if "lane" not in existing_cols:
                    connection.execute("ALTER TABLE usage_events ADD COLUMN lane TEXT")
                if "forced_exit" not in existing_cols:
                    connection.execute(
                        "ALTER TABLE usage_events ADD COLUMN forced_exit INTEGER NOT NULL DEFAULT 0"
                    )

            _create_all_tables_idempotent(connection)

            if has_usage_events:
                # Backfill UserProfileStore tables from legacy v1 usage_events
                connection.execute("DELETE FROM profile_file_opens")
                connection.execute("DELETE FROM profile_extension_counts")
                connection.execute("DELETE FROM profile_topic_counts")

                connection.execute(
                    """
                    INSERT INTO profile_file_opens (document_id, open_count, last_opened_at)
                    SELECT document_id, COUNT(*), COALESCE(MAX(created_at), ?)
                    FROM usage_events
                    WHERE event_type = 'open' AND document_id IS NOT NULL
                    GROUP BY document_id
                    """,
                    (_now_iso(),),
                )
                connection.execute(
                    """
                    INSERT INTO profile_extension_counts (extension, open_count)
                    SELECT extension, COUNT(*)
                    FROM usage_events
                    WHERE event_type = 'open' AND extension IS NOT NULL
                    GROUP BY extension
                    """
                )
                connection.execute(
                    """
                    INSERT INTO profile_topic_counts (topic, topic_count)
                    SELECT topic, COUNT(*)
                    FROM usage_events
                    WHERE event_type = 'topic' AND topic IS NOT NULL
                    GROUP BY topic
                    """
                )

            if _migration_fail_hook is not None:
                _migration_fail_hook(connection)

            connection.execute(
                """
                INSERT OR REPLACE INTO schema_migrations (version, applied_at, description)
                VALUES (?, ?, ?)
                """,
                (
                    CURRENT_SCHEMA_VERSION,
                    _now_iso(),
                    "Separate UserProfileStore, MetadataIndex, and VectorIndex tables; add search telemetry columns",
                ),
            )
            connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
            return CURRENT_SCHEMA_VERSION
    finally:
        connection.close()


class UserProfileStore:
    """Dedicated store for aggregated user file-open, extension, and topic preferences.

    Does not store raw filesystem paths or document contents.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_database_schema(self.db_path)

    def record_open(
        self,
        document_id: str,
        extension: str | None = None,
        topics: tuple[str, ...] = (),
        *,
        opened_at: str | None = None,
    ) -> None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        timestamp = opened_at or _now_iso()
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                self._apply_open_in_transaction(
                    connection,
                    document_id=document_id,
                    extension=extension,
                    topics=topics,
                    timestamp=timestamp,
                )
        finally:
            connection.close()

    @staticmethod
    def _apply_open_in_transaction(
        connection: sqlite3.Connection,
        *,
        document_id: str,
        extension: str | None,
        topics: Sequence[str],
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO profile_file_opens (document_id, open_count, last_opened_at)
            VALUES (?, 1, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                open_count = profile_file_opens.open_count + 1,
                last_opened_at = excluded.last_opened_at
            """,
            (document_id, timestamp),
        )
        if extension is not None:
            connection.execute(
                """
                INSERT INTO profile_extension_counts (extension, open_count)
                VALUES (?, 1)
                ON CONFLICT(extension) DO UPDATE SET
                    open_count = profile_extension_counts.open_count + 1
                """,
                (extension,),
            )
        for topic in topics or ():
            if topic is not None:
                connection.execute(
                    """
                    INSERT INTO profile_topic_counts (topic, topic_count)
                    VALUES (?, 1)
                    ON CONFLICT(topic) DO UPDATE SET
                        topic_count = profile_topic_counts.topic_count + 1
                    """,
                    (topic,),
                )

    def load_profile(self) -> UserProfile:
        connection = _connect(self.db_path)
        try:
            file_opens = dict(
                connection.execute(
                    "SELECT document_id, open_count FROM profile_file_opens WHERE open_count > 0"
                ).fetchall()
            )
            preferred_extensions = dict(
                connection.execute(
                    "SELECT extension, open_count FROM profile_extension_counts WHERE open_count > 0"
                ).fetchall()
            )
            topic_interests = dict(
                connection.execute(
                    "SELECT topic, topic_count FROM profile_topic_counts WHERE topic_count > 0"
                ).fetchall()
            )
        finally:
            connection.close()

        return UserProfile(
            file_opens={str(key): int(value) for key, value in file_opens.items()},
            preferred_extensions={
                str(key): int(value) for key, value in preferred_extensions.items()
            },
            topic_interests={str(key): int(value) for key, value in topic_interests.items()},
        )

    def clear(self) -> None:
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                connection.execute("DELETE FROM profile_file_opens")
                connection.execute("DELETE FROM profile_extension_counts")
                connection.execute("DELETE FROM profile_topic_counts")
        finally:
            connection.close()


class UsageLog:
    """Append-only usage event log with backward-compatible UserProfileStore delegation."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_database_schema(self.db_path)
        self.profile_store = UserProfileStore(self.db_path)

    def schema_version(self) -> int:
        connection = _connect(self.db_path)
        try:
            row = connection.execute("PRAGMA user_version").fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    def record_search(
        self,
        raw_query: str,
        result_count: int,
        *,
        lane: str | None = None,
        forced_exit: bool = False,
    ) -> None:
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                connection.execute(
                    """
                    INSERT INTO usage_events
                        (event_type, raw_query, result_count, lane, forced_exit, created_at)
                    VALUES
                        (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "search",
                        raw_query,
                        result_count,
                        lane,
                        1 if forced_exit else 0,
                        self._now(),
                    ),
                )
        finally:
            connection.close()

    def record_open(self, document: FileDocument, topics: tuple[str, ...] = ()) -> None:
        topics = tuple(topics or ())
        now = self._now()
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                connection.execute(
                    """
                    INSERT INTO usage_events
                        (event_type, document_id, path, extension, created_at)
                    VALUES
                        (?, ?, ?, ?, ?)
                    """,
                    ("open", document.id, str(document.path), document.extension, now),
                )
                for topic in topics:
                    connection.execute(
                        """
                        INSERT INTO usage_events
                            (event_type, document_id, path, extension, topic, created_at)
                        VALUES
                            (?, ?, ?, ?, ?, ?)
                        """,
                        ("topic", document.id, str(document.path), document.extension, topic, now),
                    )
                UserProfileStore._apply_open_in_transaction(
                    connection,
                    document_id=document.id,
                    extension=document.extension,
                    topics=topics,
                    timestamp=now,
                )
        finally:
            connection.close()

    def load_profile(self) -> UserProfile:
        return self.profile_store.load_profile()

    def _connect(self) -> sqlite3.Connection:
        return _connect(self.db_path)

    def _now(self) -> str:
        return _now_iso()


class MetadataIndex:
    """Persistent document metadata catalog.

    Stores document IDs, filenames, extensions, timestamps, sizes, content hashes,
    and local resolved paths (sensitive local metadata for file opening).
    Never stores raw document contents or user activity history.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_database_schema(self.db_path)

    def sync_documents(
        self,
        documents: Sequence[FileDocument],
        *,
        prune_missing: bool = True,
    ) -> int:
        now = _now_iso()
        rows: list[tuple[object, ...]] = []
        valid_ids: set[str] = set()

        for doc in documents:
            valid_ids.add(doc.id)
            try:
                resolved_path = str(doc.path.resolve())
                stat_size = doc.path.stat().st_size if not doc.is_directory else 0
            except OSError:
                resolved_path = str(doc.path)
                stat_size = len((doc.content or "").encode("utf-8", errors="ignore"))
            rows.append(
                (
                    doc.id,
                    resolved_path,
                    str(doc.path.parent),
                    doc.path.name,
                    doc.title,
                    doc.extension,
                    1 if doc.is_directory else 0,
                    int(stat_size),
                    doc.modified_at.isoformat() if doc.modified_at is not None else None,
                    doc.created_at.isoformat() if doc.created_at is not None else None,
                    compute_document_content_hash(doc),
                    now,
                )
            )

        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                if rows:
                    connection.executemany(
                        """
                        INSERT INTO metadata_documents (
                            document_id,
                            resolved_path,
                            parent_dir,
                            name,
                            title,
                            extension,
                            is_directory,
                            size_bytes,
                            modified_at,
                            created_at,
                            content_hash,
                            indexed_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id) DO UPDATE SET
                            resolved_path = excluded.resolved_path,
                            parent_dir = excluded.parent_dir,
                            name = excluded.name,
                            title = excluded.title,
                            extension = excluded.extension,
                            is_directory = excluded.is_directory,
                            size_bytes = excluded.size_bytes,
                            modified_at = excluded.modified_at,
                            created_at = excluded.created_at,
                            content_hash = excluded.content_hash,
                            indexed_at = excluded.indexed_at
                        """,
                        rows,
                    )
                if prune_missing:
                    if not valid_ids:
                        connection.execute("DELETE FROM metadata_documents")
                    else:
                        placeholders = ",".join("?" for _ in valid_ids)
                        connection.execute(
                            f"DELETE FROM metadata_documents WHERE document_id NOT IN ({placeholders})",
                            tuple(sorted(valid_ids)),
                        )
        finally:
            connection.close()
        return len(rows)

    def upsert_documents(self, documents: Sequence[FileDocument]) -> int:
        return self.sync_documents(documents, prune_missing=False)

    def remove_missing(self, valid_document_ids: Sequence[str] | set[str]) -> int:
        valid_set = set(valid_document_ids)
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                if not valid_set:
                    cursor = connection.execute("DELETE FROM metadata_documents")
                else:
                    placeholders = ",".join("?" for _ in valid_set)
                    cursor = connection.execute(
                        f"DELETE FROM metadata_documents WHERE document_id NOT IN ({placeholders})",
                        tuple(sorted(valid_set)),
                    )
                return int(cursor.rowcount or 0)
        finally:
            connection.close()

    def delete_document(self, document_id: str) -> None:
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                connection.execute(
                    "DELETE FROM metadata_documents WHERE document_id = ?",
                    (document_id,),
                )
        finally:
            connection.close()

    def get(self, document_id: str) -> StoredDocumentMetadata | None:
        connection = _connect(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    resolved_path,
                    parent_dir,
                    name,
                    title,
                    extension,
                    is_directory,
                    size_bytes,
                    modified_at,
                    created_at,
                    content_hash,
                    indexed_at
                FROM metadata_documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return self._row_to_metadata(row)

    def list_documents(self) -> tuple[StoredDocumentMetadata, ...]:
        connection = _connect(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT
                    document_id,
                    resolved_path,
                    parent_dir,
                    name,
                    title,
                    extension,
                    is_directory,
                    size_bytes,
                    modified_at,
                    created_at,
                    content_hash,
                    indexed_at
                FROM metadata_documents
                ORDER BY document_id ASC
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(self._row_to_metadata(row) for row in rows)

    def matching_document_ids(
        self,
        *,
        file_types: Sequence[str] = (),
        date_window: DateWindow | None = None,
        filename_hints: Sequence[str] = (),
    ) -> set[str]:
        entries = self.list_documents()
        matched: set[str] = set()
        type_set = {ft.lower().removeprefix(".") for ft in file_types if ft}
        hint_list = [hint.lower() for hint in filename_hints if hint]

        for entry in entries:
            ext_ok = bool(type_set and entry.extension.lower() in type_set)
            date_ok = bool(
                date_window is not None
                and entry.modified_at is not None
                and date_window.contains(entry.modified_at)
            )
            hint_ok = bool(hint_list and any(h in entry.name.lower() for h in hint_list))
            if ext_ok or date_ok or hint_ok:
                matched.add(entry.document_id)
        return matched

    @staticmethod
    def _row_to_metadata(row: tuple[object, ...]) -> StoredDocumentMetadata:
        mod_str = str(row[8]) if row[8] is not None else None
        cre_str = str(row[9]) if row[9] is not None else None
        return StoredDocumentMetadata(
            document_id=str(row[0]),
            resolved_path=str(row[1]),
            parent_dir=str(row[2]),
            name=str(row[3]),
            title=str(row[4]),
            extension=str(row[5]),
            is_directory=bool(row[6]),
            size_bytes=int(row[7]),
            modified_at=datetime.fromisoformat(mod_str) if mod_str else None,
            created_at=datetime.fromisoformat(cre_str) if cre_str else None,
            content_hash=str(row[10]),
            indexed_at=str(row[11]),
        )


class VectorIndex:
    """Persistent SQLite dense vector index scoped to a single (model_id, dimensions).

    Stores only document_id, content_hash, model_id, dimensions, and IEEE-754 float64
    vector blobs. Never persists file paths or document text.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_database_schema(self.db_path)

    def active_model_info(self) -> tuple[str, int] | None:
        connection = _connect(self.db_path)
        try:
            row = connection.execute(
                "SELECT model_id, dimensions FROM vector_index_meta WHERE singleton_id = 1"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return str(row[0]), int(row[1])

    def count(self) -> int:
        connection = _connect(self.db_path)
        try:
            row = connection.execute("SELECT COUNT(*) FROM document_embeddings").fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    def has_document(self, document_id: str) -> bool:
        connection = _connect(self.db_path)
        try:
            row = connection.execute(
                "SELECT 1 FROM document_embeddings WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return row is not None
        finally:
            connection.close()

    def index_documents(
        self,
        documents: Sequence[FileDocument],
        provider: EmbeddingProvider,
        *,
        prune_missing: bool = False,
        reset_on_model_change: bool = False,
        max_input_chars: int | None = None,
        batch_size: int | None = None,
    ) -> int:
        """Index new or modified documents atomically after validating all embedding batches.

        Returns the number of documents embedded and written in this call.
        """
        model_id = getattr(provider, "model_id", "")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("provider.model_id must be a non-empty string")
        model_id = model_id.strip()

        provider_dims = getattr(provider, "dimensions", None)
        if provider_dims is not None:
            if isinstance(provider_dims, bool) or not isinstance(provider_dims, int) or provider_dims < 1:
                raise ValueError(f"provider.dimensions must be a positive int or None, got {provider_dims!r}")

        provider_config = getattr(provider, "config", None)
        effective_max_chars = (
            max_input_chars
            if max_input_chars is not None
            else getattr(provider_config, "max_input_chars", 4000)
        )
        effective_batch_size = (
            batch_size
            if batch_size is not None
            else getattr(provider_config, "batch_size", 16)
        )
        if isinstance(effective_batch_size, bool) or not isinstance(effective_batch_size, int) or effective_batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {effective_batch_size!r}")

        # 1. Read existing index metadata and stored content hashes before calling provider
        connection = _connect(self.db_path)
        try:
            meta_row = connection.execute(
                "SELECT model_id, dimensions FROM vector_index_meta WHERE singleton_id = 1"
            ).fetchone()
            existing_rows = connection.execute(
                "SELECT document_id, content_hash, model_id, dimensions FROM document_embeddings"
            ).fetchall()
        finally:
            connection.close()

        expected_dim: int | None = provider_dims
        model_changed = False
        if meta_row is not None:
            stored_model_id = str(meta_row[0])
            stored_dim = int(meta_row[1])
            if stored_model_id != model_id or (
                provider_dims is not None and provider_dims != stored_dim
            ):
                if not reset_on_model_change:
                    raise EmbeddingDimensionMismatchError(
                        f"VectorIndex is bound to model '{stored_model_id}' (dimensions={stored_dim}); "
                        f"cannot mix with provider model '{model_id}' (dimensions={provider_dims}). "
                        f"Pass reset_on_model_change=True to rebuild the index."
                    )
                model_changed = True
            else:
                expected_dim = stored_dim

        existing_by_id: dict[str, tuple[str, str, int]] = {}
        if not model_changed:
            for row in existing_rows:
                existing_by_id[str(row[0])] = (str(row[1]), str(row[2]), int(row[3]))

        # 2. Identify documents whose content_hash or model_id changed
        to_embed: list[tuple[FileDocument, str, str]] = []
        valid_ids: set[str] = set()
        for doc in documents:
            valid_ids.add(doc.id)
            c_hash = compute_document_content_hash(doc)
            existing = existing_by_id.get(doc.id)
            if existing is not None:
                prev_hash, prev_model, prev_dim = existing
                if (
                    prev_hash == c_hash
                    and prev_model == model_id
                    and (expected_dim is None or prev_dim == expected_dim)
                ):
                    continue
            embed_text = build_document_embedding_text(doc, max_input_chars=effective_max_chars)
            to_embed.append((doc, c_hash, embed_text))

        # 3. Call provider in bounded batches and validate EVERY batch in memory before writing
        now = _now_iso()
        prepared_rows: list[tuple[str, str, str, int, bytes, str]] = []

        for start in range(0, len(to_embed), effective_batch_size):
            batch_items = to_embed[start : start + effective_batch_size]
            batch_texts = [item[2] for item in batch_items]
            raw_vectors = provider.embed_texts(batch_texts)
            if (
                isinstance(raw_vectors, (str, bytes))
                or not isinstance(raw_vectors, Sequence)
                or len(raw_vectors) != len(batch_items)
            ):
                raise EmbeddingResponseFormatError(
                    f"Provider '{model_id}' returned {len(raw_vectors) if isinstance(raw_vectors, Sequence) else type(raw_vectors).__name__} "
                    f"vectors for batch of {len(batch_items)} documents"
                )
            for (doc, c_hash, _), raw_vec in zip(batch_items, raw_vectors):
                normalized_vec = validate_and_normalize_vector(
                    raw_vec,
                    expected_dimensions=expected_dim,
                    normalize=True,
                )
                if expected_dim is None:
                    expected_dim = len(normalized_vec)
                blob = struct.pack(f"<{expected_dim}d", *normalized_vec)
                prepared_rows.append((doc.id, c_hash, model_id, expected_dim, blob, now))

        # 4. Single atomic write transaction for metadata, upserts, and optional pruning
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                current_meta = connection.execute(
                    "SELECT model_id, dimensions FROM vector_index_meta WHERE singleton_id = 1"
                ).fetchone()
                if current_meta is not None:
                    cur_model_id = str(current_meta[0])
                    cur_dim = int(current_meta[1])
                    if cur_model_id != model_id or (
                        expected_dim is not None and cur_dim != expected_dim
                    ):
                        if not reset_on_model_change:
                            raise EmbeddingDimensionMismatchError(
                                f"Concurrent model/dimension mismatch in VectorIndex: "
                                f"stored=('{cur_model_id}', {cur_dim}) vs new=('{model_id}', {expected_dim})"
                            )
                        connection.execute("DELETE FROM document_embeddings")

                if expected_dim is not None:
                    connection.execute(
                        """
                        INSERT INTO vector_index_meta (singleton_id, model_id, dimensions, updated_at)
                        VALUES (1, ?, ?, ?)
                        ON CONFLICT(singleton_id) DO UPDATE SET
                            model_id = excluded.model_id,
                            dimensions = excluded.dimensions,
                            updated_at = excluded.updated_at
                        """,
                        (model_id, expected_dim, now),
                    )

                if prepared_rows:
                    connection.executemany(
                        """
                        INSERT INTO document_embeddings (
                            document_id,
                            content_hash,
                            model_id,
                            dimensions,
                            vector_blob,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(document_id) DO UPDATE SET
                            content_hash = excluded.content_hash,
                            model_id = excluded.model_id,
                            dimensions = excluded.dimensions,
                            vector_blob = excluded.vector_blob,
                            updated_at = excluded.updated_at
                        """,
                        prepared_rows,
                    )

                if prune_missing:
                    if not valid_ids:
                        connection.execute("DELETE FROM document_embeddings")
                    else:
                        placeholders = ",".join("?" for _ in valid_ids)
                        connection.execute(
                            f"DELETE FROM document_embeddings WHERE document_id NOT IN ({placeholders})",
                            tuple(sorted(valid_ids)),
                        )
        finally:
            connection.close()

        return len(prepared_rows)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        model_id: str,
        limit: int | None = 10,
        candidate_ids: set[str] | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Compute deterministic cosine similarity rankings against stored vectors."""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        model_id = model_id.strip()

        normalized_query = validate_and_normalize_vector(query_vector, normalize=True)
        query_dim = len(normalized_query)

        connection = _connect(self.db_path)
        try:
            meta_row = connection.execute(
                "SELECT model_id, dimensions FROM vector_index_meta WHERE singleton_id = 1"
            ).fetchone()
            if meta_row is None:
                return ()
            stored_model_id = str(meta_row[0])
            stored_dim = int(meta_row[1])
            if stored_model_id != model_id:
                raise EmbeddingDimensionMismatchError(
                    f"VectorIndex model '{stored_model_id}' does not match query model '{model_id}'"
                )
            if stored_dim != query_dim:
                raise EmbeddingDimensionMismatchError(
                    f"VectorIndex dimension ({stored_dim}) does not match query vector dimension ({query_dim})"
                )

            rows = connection.execute(
                "SELECT document_id, model_id, dimensions, vector_blob FROM document_embeddings"
            ).fetchall()
        finally:
            connection.close()

        if not rows:
            return ()

        expected_blob_len = stored_dim * 8
        unpack_fmt = f"<{stored_dim}d"
        scored: list[tuple[str, float]] = []

        for row in rows:
            doc_id = str(row[0])
            if candidate_ids is not None and doc_id not in candidate_ids:
                continue
            row_model = str(row[1])
            row_dim = int(row[2])
            blob = bytes(row[3])
            if row_model != model_id or row_dim != stored_dim or len(blob) != expected_blob_len:
                raise EmbeddingDimensionMismatchError(
                    f"Stored vector row for '{doc_id}' is inconsistent with index metadata ('{stored_model_id}', dim={stored_dim})"
                )
            doc_vec = struct.unpack(unpack_fmt, blob)
            dot = sum(q_val * d_val for q_val, d_val in zip(normalized_query, doc_vec))
            if not math.isfinite(dot):
                continue
            score = clamp_score(dot)
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        if limit is None:
            return tuple(scored)
        return tuple(scored[:limit])

    def delete_document(self, document_id: str) -> None:
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                connection.execute(
                    "DELETE FROM document_embeddings WHERE document_id = ?",
                    (document_id,),
                )
        finally:
            connection.close()

    def clear(self) -> None:
        connection = _connect(self.db_path)
        try:
            with _immediate_transaction(connection):
                connection.execute("DELETE FROM document_embeddings")
                connection.execute("DELETE FROM vector_index_meta")
        finally:
            connection.close()
