"""Persistent usage log and profile store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from scoutx.models import FileDocument, UserProfile


class UsageLog:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record_search(self, raw_query: str, result_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO usage_events
                    (event_type, raw_query, result_count, created_at)
                VALUES
                    (?, ?, ?, ?)
                """,
                ("search", raw_query, result_count, self._now()),
            )

    def record_open(self, document: FileDocument, topics: tuple[str, ...] = ()) -> None:
        topics = topics or tuple()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO usage_events
                    (event_type, document_id, path, extension, created_at)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                ("open", document.id, str(document.path), document.extension, self._now()),
            )
            for topic in topics:
                connection.execute(
                    """
                    INSERT INTO usage_events
                        (event_type, document_id, path, extension, topic, created_at)
                    VALUES
                        (?, ?, ?, ?, ?, ?)
                    """,
                    ("topic", document.id, str(document.path), document.extension, topic, self._now()),
                )

    def load_profile(self) -> UserProfile:
        with self._connect() as connection:
            file_opens = dict(
                connection.execute(
                    """
                    SELECT document_id, COUNT(*)
                    FROM usage_events
                    WHERE event_type = 'open' AND document_id IS NOT NULL
                    GROUP BY document_id
                    """
                ).fetchall()
            )
            preferred_extensions = dict(
                connection.execute(
                    """
                    SELECT extension, COUNT(*)
                    FROM usage_events
                    WHERE event_type = 'open' AND extension IS NOT NULL
                    GROUP BY extension
                    """
                ).fetchall()
            )
            topic_interests = dict(
                connection.execute(
                    """
                    SELECT topic, COUNT(*)
                    FROM usage_events
                    WHERE event_type = 'topic' AND topic IS NOT NULL
                    GROUP BY topic
                    """
                ).fetchall()
            )

        return UserProfile(
            file_opens={str(key): int(value) for key, value in file_opens.items()},
            preferred_extensions={str(key): int(value) for key, value in preferred_extensions.items()},
            topic_interests={str(key): int(value) for key, value in topic_interests.items()},
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
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

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")
