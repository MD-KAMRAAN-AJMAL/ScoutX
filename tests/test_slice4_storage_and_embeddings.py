"""Deterministic unit and loopback integration tests for Slice 4:
embeddings, persistent SQLite indices, storage separation/migration, and bounded ingestion.
"""

from __future__ import annotations

import io
import json
import math
import os
import sqlite3
import tempfile
import threading
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

from scoutx.cli import main as cli_main
from scoutx.document_loader import DocumentLoader
from scoutx.embeddings import (
    EmbeddingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingResponseFormatError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
    GeminiEmbeddingProvider,
    LocalHashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    bound_embedding_input,
    build_document_embedding_text,
    create_embedding_provider,
    validate_and_normalize_vector,
)
from scoutx.engine import ScoutXEngine
from scoutx.models import FileDocument, RetrievalTool, UserProfile
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent
from scoutx.storage import (
    CURRENT_SCHEMA_VERSION,
    MetadataIndex,
    StorageSchemaError,
    UsageLog,
    UserProfileStore,
    VectorIndex,
    compute_document_content_hash,
    ensure_database_schema,
)


def _create_docx_bytes(paragraphs: Sequence[str]) -> bytes:
    buf = io.BytesIO()
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml_bytes = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    ).encode("utf-8")
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml_bytes)
    return buf.getvalue()


class Slice4StorageAndEmbeddingsTests(unittest.TestCase):
    def test_01_local_hash_embedding_repeatability_normalization_and_documented_limitations(
        self,
    ) -> None:
        config_64 = EmbeddingConfig(
            provider="local_hash",
            model="scoutx-feature-hash-v1",
            dimensions=64,
        )
        provider_a = LocalHashEmbeddingProvider(config_64)
        provider_b = LocalHashEmbeddingProvider(config_64)

        self.assertEqual(provider_a.model_id, "local_hash:scoutx-feature-hash-v1:d64")
        self.assertEqual(provider_a.dimensions, 64)

        vec_1 = provider_a.embed_query("Enterprise pricing forecast and budget sheet")
        vec_2 = provider_b.embed_query("Enterprise pricing forecast and budget sheet")
        self.assertEqual(vec_1, vec_2)
        self.assertEqual(len(vec_1), 64)
        norm = math.sqrt(sum(x * x for x in vec_1))
        self.assertAlmostEqual(norm, 1.0, places=9)

        # Changing dimensions changes model_id so different dimension spaces cannot mix
        provider_128 = LocalHashEmbeddingProvider(
            EmbeddingConfig(provider="local_hash", dimensions=128)
        )
        self.assertEqual(provider_128.model_id, "local_hash:scoutx-feature-hash-v1:d128")
        self.assertEqual(len(provider_128.embed_query("pricing")), 128)

        # Representative retrieval behavior: morphological + curated synonym overlap ranks above unrelated text
        q_vec = provider_a.embed_query("pricing forecast")
        doc_related = provider_a.embed_query("price forecasting budget revenue")
        doc_unrelated = provider_a.embed_query("kernel panic interrupt stacktrace")
        sim_related = sum(a * b for a, b in zip(q_vec, doc_related))
        sim_unrelated = sum(a * b for a, b in zip(q_vec, doc_unrelated))
        self.assertGreater(sim_related, 0.35)
        self.assertGreater(sim_related, sim_unrelated + 0.25)

        # Documented limitation 1: bag-of-features ignores negation ("approved budget" vs "not approved budget")
        v_pos = provider_a.embed_query("approved budget proposal")
        v_neg = provider_a.embed_query("not approved budget proposal")
        sim_negation = sum(a * b for a, b in zip(v_pos, v_neg))
        self.assertGreater(sim_negation, 0.70)

        # Empty / punctuation-only text raises EmbeddingResponseFormatError
        with self.assertRaises(EmbeddingResponseFormatError):
            provider_a.embed_query("   !!! ---   ")

    def test_02_embedding_config_per_input_max_chars_batching_and_secret_redaction(
        self,
    ) -> None:
        # Invalid configurations
        with self.assertRaises(ValueError):
            EmbeddingConfig(provider="unknown")
        with self.assertRaises(ValueError):
            EmbeddingConfig(provider="local_hash", dimensions=4)
        with self.assertRaises(ValueError):
            EmbeddingConfig(provider="local_hash", dimensions=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EmbeddingConfig(provider="openai_compatible", api_key_env="sk-live-secret-12345")
        with self.assertRaises(ValueError):
            EmbeddingConfig(provider="local_hash", max_input_chars=10)
        with self.assertRaises(ValueError):
            EmbeddingConfig(provider="local_hash", timeout_seconds=float("inf"))

        # Verify build_document_embedding_text strips paths and enforces per-input max_input_chars
        secret_dir = Path("/Users/alice/PrivateVault/SecretProject")
        doc = FileDocument(
            id="doc1",
            path=secret_dir / "roadmap_2026.md",
            title="roadmap_2026",
            extension="md",
            content="quarterly milestones " * 100,
            metadata={
                "name": "roadmap_2026.md",
                "extension": "md",
                "kind": "file",
                "parent": str(secret_dir),
                "path": str(secret_dir / "roadmap_2026.md"),
            },
        )
        doc_text = build_document_embedding_text(doc, max_input_chars=120)
        self.assertLessEqual(len(doc_text), 120)
        self.assertNotIn("PrivateVault", doc_text)
        self.assertNotIn("SecretProject", doc_text)
        self.assertNotIn("/Users/alice", doc_text)

        # Loopback server capturing requests for OpenAI-compatible and Gemini providers
        captured_requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        mode = {"status": 200}

        class _EmbeddingLoopbackHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                parsed_body = json.loads(raw_body)
                headers_Snapshot = {k.lower(): v for k, v in self.headers.items()}
                captured_requests.append((self.path, headers_Snapshot, parsed_body))

                if mode["status"] == 500:
                    err = b'{"error": "upstream failure for key SUPER_SECRET_EMB_KEY_999"}'
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return

                if self.path.endswith("/embeddings"):
                    inputs = parsed_body["input"]
                    data = [
                        {"index": idx, "embedding": [0.6, 0.8, 0.0, 0.0]}
                        for idx in range(len(inputs))
                    ]
                    resp = json.dumps({"data": data}).encode("utf-8")
                else:
                    reqs = parsed_body["requests"]
                    embeddings = [
                        {"values": [0.0, 0.6, 0.8, 0.0]} for _ in range(len(reqs))
                    ]
                    resp = json.dumps({"embeddings": embeddings}).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), _EmbeddingLoopbackHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(
                os.environ,
                {"TEST_EMB_KEY": "SUPER_SECRET_EMB_KEY_999"},
                clear=False,
            ):
                # 1. OpenAI-compatible batching + per-input max_input_chars enforcement
                oai_cfg = EmbeddingConfig(
                    provider="openai_compatible",
                    model="text-embedding-test",
                    dimensions=4,
                    base_url=f"http://127.0.0.1:{port}/v1",
                    api_key_env="TEST_EMB_KEY",
                    max_input_chars=64,
                    batch_size=2,
                )
                oai_provider = create_embedding_provider(oai_cfg)
                self.assertIsInstance(oai_provider, OpenAICompatibleEmbeddingProvider)

                long_inputs = [f"item-{i}-" + ("x" * 200) for i in range(5)]
                vectors = oai_provider.embed_texts(long_inputs)
                self.assertEqual(len(vectors), 5)
                # 5 items with batch_size=2 -> 3 HTTP requests (2, 2, 1)
                self.assertEqual(len(captured_requests), 3)
                batch_lengths = [len(req[2]["input"]) for req in captured_requests]
                self.assertEqual(batch_lengths, [2, 2, 1])
                for _, _, body in captured_requests:
                    for item_text in body["input"]:
                        self.assertLessEqual(len(item_text), 64)

                # 2. Gemini batching + per-input max_input_chars enforcement
                captured_requests.clear()
                gem_cfg = EmbeddingConfig(
                    provider="gemini",
                    model="text-embedding-004",
                    dimensions=4,
                    base_url=f"http://127.0.0.1:{port}/v1beta",
                    api_key_env="TEST_EMB_KEY",
                    max_input_chars=50,
                    batch_size=2,
                )
                gem_provider = create_embedding_provider(gem_cfg)
                self.assertIsInstance(gem_provider, GeminiEmbeddingProvider)
                gem_vectors = gem_provider.embed_texts(long_inputs[:3])
                self.assertEqual(len(gem_vectors), 3)
                self.assertEqual(len(captured_requests), 2)
                for _, headers, body in captured_requests:
                    self.assertEqual(headers.get("x-goog-api-key"), "SUPER_SECRET_EMB_KEY_999")
                    for req_item in body["requests"]:
                        part_text = req_item["content"]["parts"][0]["text"]
                        self.assertLessEqual(len(part_text), 50)

                # 3. Secret redaction on HTTP 500
                mode["status"] = 500
                with self.assertRaises(EmbeddingUnavailableError) as ctx:
                    oai_provider.embed_query("test query")
                self.assertNotIn("SUPER_SECRET_EMB_KEY_999", str(ctx.exception))
                self.assertIn("[REDACTED]", str(ctx.exception))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_03_vector_validation_and_all_or_nothing_batch_writes(self) -> None:
        # Direct vector validation checks
        with self.assertRaises(EmbeddingResponseFormatError):
            validate_and_normalize_vector([])
        with self.assertRaises(EmbeddingResponseFormatError):
            validate_and_normalize_vector([True, False])
        with self.assertRaises(EmbeddingResponseFormatError):
            validate_and_normalize_vector([1.0, float("nan")])
        with self.assertRaises(EmbeddingResponseFormatError):
            validate_and_normalize_vector([1.0, float("inf")])
        with self.assertRaises(EmbeddingResponseFormatError):
            validate_and_normalize_vector([0.0, 0.0, 0.0])
        with self.assertRaises(EmbeddingDimensionMismatchError):
            validate_and_normalize_vector([1.0, 0.0], expected_dimensions=4)

        # All-or-nothing batch write test on VectorIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "index.sqlite"
            vector_index = VectorIndex(db_path)

            docs = tuple(
                FileDocument(
                    id=f"doc_{i}",
                    path=Path(tmpdir) / f"file_{i}.txt",
                    title=f"file_{i}",
                    extension="txt",
                    content=f"content for document number {i}",
                )
                for i in range(4)
            )

            class _FlakyBatchProvider:
                model_id = "custom:flaky-v1"
                dimensions = 4

                def __init__(self) -> None:
                    self.calls = 0
                    self.fail_on_second_batch = True

                def embed_query(self, text: str) -> tuple[float, ...]:
                    return (0.6, 0.8, 0.0, 0.0)

                def embed_texts(
                    self, texts: Sequence[str]
                ) -> tuple[tuple[float, ...], ...]:
                    self.calls += 1
                    if self.fail_on_second_batch and self.calls == 2:
                        # Return a zero-norm vector on batch 2 to trigger failure after batch 1 succeeded
                        return tuple((0.0, 0.0, 0.0, 0.0) for _ in texts)
                    return tuple((0.6, 0.8, 0.0, 0.0) for _ in texts)

            provider = _FlakyBatchProvider()
            with self.assertRaises(EmbeddingResponseFormatError):
                vector_index.index_documents(docs, provider, batch_size=2)

            # Batch 1 must NOT have been persisted, and vector_index_meta must remain unset
            self.assertEqual(vector_index.count(), 0)
            self.assertIsNone(vector_index.active_model_info())

            # Retry after fixing provider succeeds cleanly and indexes all 4 documents
            provider.fail_on_second_batch = False
            written = vector_index.index_documents(docs, provider, batch_size=2)
            self.assertEqual(written, 4)
            self.assertEqual(vector_index.count(), 4)
            self.assertEqual(vector_index.active_model_info(), ("custom:flaky-v1", 4))

    def test_04_vector_index_model_mixing_guard_and_reopen_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "vectors.sqlite"
            docs = (
                FileDocument(
                    id="doc_alpha",
                    path=Path(tmpdir) / "alpha.md",
                    title="alpha",
                    extension="md",
                    content="enterprise pricing forecast revenue sheet",
                ),
                FileDocument(
                    id="doc_beta",
                    path=Path(tmpdir) / "beta.md",
                    title="beta",
                    extension="md",
                    content="weekly engineering standup meeting notes",
                ),
            )

            provider_64 = LocalHashEmbeddingProvider(
                EmbeddingConfig(provider="local_hash", model="hash-v1", dimensions=64)
            )
            index_1 = VectorIndex(db_path)
            written_1 = index_1.index_documents(docs, provider_64)
            self.assertEqual(written_1, 2)

            q_vec = provider_64.embed_query("pricing forecast")
            scores_before_reopen = index_1.search(q_vec, model_id=provider_64.model_id)

            # Reopen SQLite VectorIndex and verify identical scores and order
            index_reopened = VectorIndex(db_path)
            self.assertEqual(
                index_reopened.active_model_info(),
                ("local_hash:hash-v1:d64", 64),
            )
            scores_after_reopen = index_reopened.search(
                q_vec, model_id=provider_64.model_id
            )
            self.assertEqual(scores_before_reopen, scores_after_reopen)
            self.assertEqual(scores_after_reopen[0][0], "doc_alpha")

            # Model-mixing guard 1: different model name with same 64 dimensions
            provider_other_name = LocalHashEmbeddingProvider(
                EmbeddingConfig(provider="local_hash", model="hash-v2", dimensions=64)
            )
            with self.assertRaises(EmbeddingDimensionMismatchError):
                index_reopened.index_documents(docs, provider_other_name)
            with self.assertRaises(EmbeddingDimensionMismatchError):
                index_reopened.search(q_vec, model_id=provider_other_name.model_id)

            # Model-mixing guard 2: different dimensions (128 vs 64)
            provider_128 = LocalHashEmbeddingProvider(
                EmbeddingConfig(provider="local_hash", model="hash-v1", dimensions=128)
            )
            with self.assertRaises(EmbeddingDimensionMismatchError):
                index_reopened.index_documents(docs, provider_128)

            # Explicit reset_on_model_change=True replaces the index cleanly
            rebuilt = index_reopened.index_documents(
                docs, provider_128, reset_on_model_change=True
            )
            self.assertEqual(rebuilt, 2)
            self.assertEqual(
                index_reopened.active_model_info(),
                ("local_hash:hash-v1:d128", 128),
            )

    def test_05_document_identity_lifecycle_unchanged_modified_renamed_moved_deleted_replaced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            root.mkdir()
            db_path = Path(tmpdir) / "state.sqlite"

            file_a = root / "budget.txt"
            file_b = root / "notes.txt"
            file_a.write_text("initial budget numbers 2026", encoding="utf-8")
            file_b.write_text("engineering sync notes", encoding="utf-8")

            loader = DocumentLoader()
            meta_index = MetadataIndex(db_path)
            vec_index = VectorIndex(db_path)
            usage_log = UsageLog(db_path)

            base_provider = LocalHashEmbeddingProvider()
            embed_calls: list[int] = []
            orig_embed_texts = base_provider.embed_texts

            def _counting_embed_texts(texts: Sequence[str]):
                embed_calls.append(len(texts))
                return orig_embed_texts(texts)

            base_provider.embed_texts = _counting_embed_texts  # type: ignore[method-assign]

            # Initial ingest
            docs_1 = loader.load_directory(root)
            self.assertEqual(len(docs_1), 2)
            doc_a_initial = next(d for d in docs_1 if d.path.name == "budget.txt")
            usage_log.record_open(doc_a_initial, topics=("finance",))

            meta_index.sync_documents(docs_1, prune_missing=True)
            written_1 = vec_index.index_documents(
                docs_1, base_provider, prune_missing=True
            )
            self.assertEqual(written_1, 2)
            self.assertEqual(sum(embed_calls), 2)

            # 1. Unchanged: 0 re-embedding calls
            embed_calls.clear()
            docs_unchanged = loader.load_directory(root)
            written_unchanged = vec_index.index_documents(
                docs_unchanged, base_provider, prune_missing=True
            )
            self.assertEqual(written_unchanged, 0)
            self.assertEqual(embed_calls, [])

            # 2. Modified in place: same document_id, new content_hash, only 1 file re-embedded
            file_a.write_text("updated budget forecast and pricing 2027", encoding="utf-8")
            docs_modified = loader.load_directory(root)
            doc_a_modified = next(d for d in docs_modified if d.path.name == "budget.txt")
            self.assertEqual(doc_a_modified.id, doc_a_initial.id)
            self.assertNotEqual(
                compute_document_content_hash(doc_a_modified),
                compute_document_content_hash(doc_a_initial),
            )
            meta_index.sync_documents(docs_modified, prune_missing=True)
            written_mod = vec_index.index_documents(
                docs_modified, base_provider, prune_missing=True
            )
            self.assertEqual(written_mod, 1)
            self.assertEqual(embed_calls, [1])

            # 3. Renamed in same directory: new document_id, old ID pruned from indices,
            # historical open count remains on old ID rather than transferring silently
            embed_calls.clear()
            renamed_a = root / "budget_renamed.txt"
            file_a.rename(renamed_a)
            docs_renamed = loader.load_directory(root)
            doc_a_renamed = next(
                d for d in docs_renamed if d.path.name == "budget_renamed.txt"
            )
            self.assertNotEqual(doc_a_renamed.id, doc_a_initial.id)
            meta_index.sync_documents(docs_renamed, prune_missing=True)
            vec_index.index_documents(docs_renamed, base_provider, prune_missing=True)
            self.assertIsNone(meta_index.get(doc_a_initial.id))
            self.assertFalse(vec_index.has_document(doc_a_initial.id))
            self.assertIsNotNone(meta_index.get(doc_a_renamed.id))
            self.assertTrue(vec_index.has_document(doc_a_renamed.id))
            profile = usage_log.load_profile()
            self.assertEqual(profile.file_opens.get(doc_a_initial.id), 1)
            self.assertIsNone(profile.file_opens.get(doc_a_renamed.id))

            # 4. Moved to subdirectory: new document_id, old renamed ID pruned
            subdir = root / "archive"
            subdir.mkdir()
            moved_a = subdir / "budget_renamed.txt"
            renamed_a.rename(moved_a)
            docs_moved = loader.load_directory(root)
            doc_a_moved = next(
                d for d in docs_moved if d.path.name == "budget_renamed.txt" and not d.is_directory
            )
            self.assertNotEqual(doc_a_moved.id, doc_a_renamed.id)
            meta_index.sync_documents(docs_moved, prune_missing=True)
            vec_index.index_documents(docs_moved, base_provider, prune_missing=True)
            self.assertFalse(vec_index.has_document(doc_a_renamed.id))
            self.assertTrue(vec_index.has_document(doc_a_moved.id))

            # 5. Replaced at same path: same document_id, new content_hash, overwrites row
            hash_before_replace = meta_index.get(doc_a_moved.id).content_hash  # type: ignore[union-attr]
            moved_a.write_text("completely different replacement document text", encoding="utf-8")
            docs_replaced = loader.load_directory(root)
            meta_index.sync_documents(docs_replaced, prune_missing=True)
            vec_index.index_documents(docs_replaced, base_provider, prune_missing=True)
            hash_after_replace = meta_index.get(doc_a_moved.id).content_hash  # type: ignore[union-attr]
            self.assertNotEqual(hash_before_replace, hash_after_replace)

            # 6. Deleted: cannot surface in RetrievalToolbox even before prune, and pruned on sync
            moved_a.unlink()
            docs_after_delete = loader.load_directory(root)
            corpus_after_delete = DocumentCorpus(docs_after_delete)
            toolbox = RetrievalToolbox(
                corpus_after_delete,
                embedding_provider=base_provider,
                vector_index=vec_index,
                metadata_index=meta_index,
            )
            parsed = QueryUnderstandingModule().parse("replacement document")
            results = toolbox.semantic_search(parsed, limit=10)
            self.assertTrue(all(c.document.id != doc_a_moved.id for c in results))
            self.assertFalse(vec_index.has_document(doc_a_moved.id))
            self.assertIsNone(meta_index.get(doc_a_moved.id))

    def test_06_legacy_v1_usage_log_migration_profile_equivalence_rollback_and_idempotency(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy_v1.sqlite"

            # 1. Construct a genuine legacy Slice 1 v1 database (user_version = 0)
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE usage_events (
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
                legacy_rows = [
                    ("search", None, None, None, None, "budget forecast", 3, "2026-10-01T10:00:00+00:00"),
                    ("open", "doc_1", "/tmp/a.csv", "csv", None, None, None, "2026-10-01T10:01:00+00:00"),
                    ("topic", "doc_1", "/tmp/a.csv", "csv", "finance", None, None, "2026-10-01T10:01:00+00:00"),
                    ("topic", "doc_1", "/tmp/a.csv", "csv", "budget", None, None, "2026-10-01T10:01:00+00:00"),
                    ("open", "doc_1", "/tmp/a.csv", "csv", None, None, None, "2026-10-02T11:00:00+00:00"),
                    ("topic", "doc_1", "/tmp/a.csv", "csv", "finance", None, None, "2026-10-02T11:00:00+00:00"),
                    ("open", "doc_2", "/tmp/b.pptx", "pptx", None, None, None, "2026-10-03T12:00:00+00:00"),
                ]
                conn.executemany(
                    """
                    INSERT INTO usage_events
                        (event_type, document_id, path, extension, topic, raw_query, result_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    legacy_rows,
                )
                conn.commit()

                # Compute pre-migration profile using exact v1 SQL queries
                v1_opens = dict(
                    conn.execute(
                        "SELECT document_id, COUNT(*) FROM usage_events WHERE event_type = 'open' AND document_id IS NOT NULL GROUP BY document_id"
                    ).fetchall()
                )
                v1_exts = dict(
                    conn.execute(
                        "SELECT extension, COUNT(*) FROM usage_events WHERE event_type = 'open' AND extension IS NOT NULL GROUP BY extension"
                    ).fetchall()
                )
                v1_topics = dict(
                    conn.execute(
                        "SELECT topic, COUNT(*) FROM usage_events WHERE event_type = 'topic' AND topic IS NOT NULL GROUP BY topic"
                    ).fetchall()
                )
                expected_profile = UserProfile(
                    file_opens={str(k): int(v) for k, v in v1_opens.items()},
                    preferred_extensions={str(k): int(v) for k, v in v1_exts.items()},
                    topic_interests={str(k): int(v) for k, v in v1_topics.items()},
                )
            finally:
                conn.close()

            # 2. Induce mid-migration failure and verify full transactional rollback
            def _boom(_conn: sqlite3.Connection) -> None:
                raise RuntimeError("induced mid-migration failure")

            with self.assertRaises(RuntimeError):
                ensure_database_schema(db_path, _migration_fail_hook=_boom)

            conn = sqlite3.connect(db_path)
            try:
                ver_after_rollback = conn.execute("PRAGMA user_version").fetchone()[0]
                self.assertEqual(ver_after_rollback, 0)
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                self.assertNotIn("profile_file_opens", tables)
                event_count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
                self.assertEqual(event_count, 7)
            finally:
                conn.close()

            # 3. Perform real migration and verify profile equivalence
            usage_log = UsageLog(db_path)
            self.assertEqual(usage_log.schema_version(), CURRENT_SCHEMA_VERSION)
            migrated_profile = usage_log.load_profile()
            self.assertEqual(migrated_profile, expected_profile)
            self.assertEqual(UserProfileStore(db_path).load_profile(), expected_profile)

            # 4. Repeat initialization across all stores; verify no double-counting or version conflict
            for _ in range(3):
                UsageLog(db_path)
                UserProfileStore(db_path)
                MetadataIndex(db_path)
                VectorIndex(db_path)
            self.assertEqual(UsageLog(db_path).load_profile(), expected_profile)
            self.assertEqual(UsageLog(db_path).schema_version(), CURRENT_SCHEMA_VERSION)

            # 5. Unsupported future schema version is rejected with StorageSchemaError
            future_db = Path(tmpdir) / "future.sqlite"
            fconn = sqlite3.connect(future_db)
            try:
                fconn.execute("PRAGMA user_version = 99")
                fconn.commit()
            finally:
                fconn.close()
            with self.assertRaises(StorageSchemaError):
                UsageLog(future_db)

    def test_07_shared_vs_separate_sqlite_files_and_storage_privacy_boundaries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "docs"
            root.mkdir()
            secret_file = root / "confidential_plan.txt"
            secret_content = "CONFIDENTIAL_BODY_TOKEN_XYZ_987654"
            secret_file.write_text(secret_content, encoding="utf-8")

            # Shared DB file case
            shared_db = Path(tmpdir) / "shared.sqlite"
            provider = LocalHashEmbeddingProvider()
            engine_shared = ScoutXEngine.from_directory(
                root,
                usage_db=shared_db,
                embedding_provider=provider,
            )
            doc = engine_shared.corpus.documents[0]
            assert engine_shared.usage_log is not None
            engine_shared.usage_log.record_open(doc, topics=("strategy",))
            resp_shared = engine_shared.search("confidential plan")
            self.assertEqual(len(resp_shared.recommendations), 1)

            # Inspect raw SQLite contents for privacy boundaries
            conn = sqlite3.connect(shared_db)
            try:
                # MetadataIndex must not contain raw file content
                meta_rows = conn.execute("SELECT * FROM metadata_documents").fetchall()
                self.assertEqual(len(meta_rows), 1)
                for row in meta_rows:
                    for col_val in row:
                        self.assertNotIn(secret_content, str(col_val))

                # VectorIndex must not contain paths or raw file content
                vec_rows = conn.execute(
                    "SELECT document_id, content_hash, model_id FROM document_embeddings"
                ).fetchall()
                self.assertEqual(len(vec_rows), 1)
                for row in vec_rows:
                    for col_val in row:
                        self.assertNotIn(secret_content, str(col_val))
                        self.assertNotIn("confidential_plan.txt", str(col_val))
                        self.assertNotIn(str(root), str(col_val))

                # UserProfileStore tables must not contain paths
                for table in (
                    "profile_file_opens",
                    "profile_extension_counts",
                    "profile_topic_counts",
                ):
                    p_rows = conn.execute(f"SELECT * FROM {table}").fetchall()
                    for row in p_rows:
                        for col_val in row:
                            self.assertNotIn(str(root), str(col_val))
                            self.assertNotIn("confidential_plan.txt", str(col_val))
            finally:
                conn.close()

            # Separate DB files case (usage_db != index_db)
            usage_only_db = Path(tmpdir) / "usage_only.sqlite"
            index_only_db = Path(tmpdir) / "index_only.sqlite"
            engine_sep = ScoutXEngine.from_directory(
                root,
                usage_db=usage_only_db,
                index_db=index_only_db,
                embedding_provider=provider,
            )
            resp_sep = engine_sep.search("confidential plan")
            self.assertEqual(len(resp_sep.recommendations), 1)
            self.assertEqual(VectorIndex(usage_only_db).count(), 0)
            self.assertEqual(VectorIndex(index_only_db).count(), 1)

    def test_08_document_loader_size_limits_bounded_office_streams_and_corrupt_archives(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # 1. Plain-text file exceeding max_file_bytes -> metadata-only FileDocument + file_too_large issue
            oversized_txt = root / "huge_log.txt"
            oversized_txt.write_text("A" * 500, encoding="utf-8")

            # 2. Valid small text file with content exceeding max_extracted_chars -> truncated
            trunc_txt = root / "long_notes.txt"
            trunc_txt.write_text("B" * 150, encoding="utf-8")

            # 3. Corrupt .docx file (raw binary bytes masquerading as .docx) -> never falls back to plain text
            corrupt_docx = root / "corrupt.docx"
            corrupt_docx.write_bytes(b"NOT_A_VALID_ZIP_ARCHIVE_SECRET_RAW_TEXT")

            # 4. Valid .docx whose uncompressed XML exceeds max_office_uncompressed_bytes
            large_xml_docx = root / "large_xml.docx"
            large_xml_docx.write_bytes(_create_docx_bytes(["word " * 200]))

            # 5. Oversized compressed Office archive exceeding max_office_archive_bytes
            big_archive_docx = root / "big_archive.docx"
            big_archive_docx.write_bytes(_create_docx_bytes(["paragraph text"]))

            # 6. Unknown extension containing NUL byte -> detected as binary
            binary_custom = root / "blob.customext"
            binary_custom.write_bytes(b"header\x00binary_payload_token")

            loader = DocumentLoader(
                max_file_bytes=200,
                max_office_archive_bytes=10_000,
                max_office_uncompressed_bytes=400,
                max_extracted_chars=100,
            )

            # Load oversized text
            doc_huge = loader.load_file(oversized_txt)
            self.assertIsNotNone(doc_huge)
            assert doc_huge is not None
            self.assertEqual(doc_huge.content, "")
            self.assertEqual(loader.last_issues[0].reason, "file_too_large")

            # Load truncated text
            doc_trunc = loader.load_file(trunc_txt)
            self.assertIsNotNone(doc_trunc)
            assert doc_trunc is not None
            self.assertEqual(len(doc_trunc.content), 100)
            self.assertEqual(loader.last_issues[0].reason, "content_truncated")

            # Load corrupt .docx -> content must be empty (no fallback to _read_plain_text!)
            doc_corrupt = loader.load_file(corrupt_docx)
            self.assertIsNotNone(doc_corrupt)
            assert doc_corrupt is not None
            self.assertEqual(doc_corrupt.content, "")
            self.assertEqual(loader.last_issues[0].reason, "corrupt_office_archive")

            # Load .docx with uncompressed XML > 400 bytes
            doc_large_xml = loader.load_file(large_xml_docx)
            self.assertIsNotNone(doc_large_xml)
            assert doc_large_xml is not None
            self.assertEqual(doc_large_xml.content, "")
            self.assertEqual(loader.last_issues[0].reason, "office_uncompressed_too_large")

            # Load .docx with compressed size > tight archive limit
            tight_archive_loader = DocumentLoader(max_office_archive_bytes=10)
            doc_big_arch = tight_archive_loader.load_file(big_archive_docx)
            self.assertIsNotNone(doc_big_arch)
            assert doc_big_arch is not None
            self.assertEqual(doc_big_arch.content, "")
            self.assertEqual(
                tight_archive_loader.last_issues[0].reason, "office_archive_too_large"
            )

            # Load unknown binary file with NUL byte
            doc_bin = loader.load_file(binary_custom)
            self.assertIsNotNone(doc_bin)
            assert doc_bin is not None
            self.assertEqual(doc_bin.content, "")
            self.assertEqual(loader.last_issues[0].reason, "binary_or_unsupported")

            # Controlled streaming-member test double where declared ZipInfo.file_size (10 bytes)
            # understates actual decompressed stream length (1000 bytes > budget 300 bytes)
            class _FakeUnderstatingArchive:
                def namelist(self) -> list[str]:
                    return ["word/document.xml"]

                def getinfo(self, name: str) -> zipfile.ZipInfo:
                    info = zipfile.ZipInfo(name)
                    info.file_size = 10  # Understated header size
                    return info

                def open(self, name: str) -> io.BytesIO:
                    return io.BytesIO(b"<w:document>" + (b"X" * 1000) + b"</w:document>")

            stream_loader = DocumentLoader(max_office_uncompressed_bytes=300)
            extracted = stream_loader._extract_office_archive_text(
                _FakeUnderstatingArchive(),  # type: ignore[arg-type]
                Path(tmpdir) / "spoofed.docx",
            )
            self.assertEqual(extracted, "")
            self.assertEqual(len(stream_loader.last_issues), 1)
            self.assertEqual(
                stream_loader.last_issues[0].reason, "office_uncompressed_too_large"
            )

    def test_09_retrieval_toolbox_dense_search_router_integration_and_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f1 = root / "Pricing_Forecast_2026.md"
            f1.write_text("enterprise pricing forecast revenue model", encoding="utf-8")
            f2 = root / "team_lunch.md"
            f2.write_text("pizza order and dietary preferences", encoding="utf-8")

            corpus = DocumentCorpus.from_directory(root)
            db_path = Path(tmpdir) / "index.sqlite"
            vec_index = VectorIndex(db_path)
            provider = LocalHashEmbeddingProvider()

            toolbox = RetrievalToolbox(
                corpus,
                embedding_provider=provider,
                vector_index=vec_index,
            )
            router = ComplexityRoutingAgent(toolbox)
            parsed = QueryUnderstandingModule().parse("pricing forecast")
            result = router.run(parsed)
            self.assertGreaterEqual(len(result.candidates), 1)
            self.assertEqual(result.candidates[0].document.id, corpus.documents[0].id)

            # Verify dense semantic search reason is populated
            sem_candidates = toolbox.semantic_search(parsed, limit=5)
            self.assertGreaterEqual(len(sem_candidates), 1)
            self.assertIn("dense semantic similarity match", sem_candidates[0].reason)

            # Simulated provider failure falls back to lexical semantic search when strict_embeddings=False
            class _BrokenEmbeddingProvider:
                model_id = "broken:v1"
                dimensions = 16

                def embed_query(self, text: str) -> tuple[float, ...]:
                    raise EmbeddingUnavailableError("embedding service offline")

                def embed_texts(
                    self, texts: Sequence[str]
                ) -> tuple[tuple[float, ...], ...]:
                    raise EmbeddingUnavailableError("embedding service offline")

            fallback_toolbox = RetrievalToolbox(
                corpus,
                embedding_provider=_BrokenEmbeddingProvider(),
                strict_embeddings=False,
            )
            fallback_candidates = fallback_toolbox.semantic_search(parsed, limit=5)
            self.assertGreaterEqual(len(fallback_candidates), 1)
            self.assertEqual(fallback_candidates[0].reason, "semantic similarity match")
            self.assertIn(
                "embedding service offline",
                fallback_toolbox.last_semantic_fallback_error or "",
            )

            # When strict_embeddings=True, error propagates and router records it in tool_errors
            strict_toolbox = RetrievalToolbox(
                corpus,
                embedding_provider=_BrokenEmbeddingProvider(),
                strict_embeddings=True,
            )
            strict_router = ComplexityRoutingAgent(strict_toolbox)
            strict_result = strict_router.run(parsed)
            self.assertIn(
                RetrievalTool.SEMANTIC,
                strict_result.trace[0].tool_errors,
            )

    def test_10_cli_embedding_and_index_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            root.mkdir()
            doc_path = root / "budget_plan.md"
            doc_path.write_text("annual budget cost forecast pricing", encoding="utf-8")
            index_db = Path(tmpdir) / "index.sqlite"

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                code = cli_main(
                    [
                        "search",
                        "budget forecast",
                        "--root",
                        str(root),
                        "--index-db",
                        str(index_db),
                        "--embedding-provider",
                        "local_hash",
                        "--embedding-dimensions",
                        "32",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("budget_plan.md", out_buf.getvalue())
            self.assertEqual(
                VectorIndex(index_db).active_model_info(),
                ("local_hash:scoutx-feature-hash-v1:d32", 32),
            )

            # Missing model for non-local_hash embedding provider returns exit code 2
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                err_code = cli_main(
                    [
                        "search",
                        "budget forecast",
                        "--root",
                        str(root),
                        "--embedding-provider",
                        "openai_compatible",
                    ]
                )
            self.assertEqual(err_code, 2)
            self.assertIn("--embedding-model", err_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
