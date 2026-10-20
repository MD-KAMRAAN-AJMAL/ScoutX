"""Deterministic regression tests for query routing calibration, lexical relevance,
media search, and graceful no-confident-match termination.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scoutx.document_loader import DocumentLoader
from scoutx.engine import ScoutXEngine
from scoutx.models import (
    FileDocument,
    ParsedQuery,
    RetrievalTool,
    SearchCandidate,
    SearchLane,
    SearchStrategy,
)
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent


class RoutingCalibrationAndRelevanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.now = datetime(2026, 3, 18, 12, 0, 0)

        # 1. Media folder and photos
        (self.root / "Pictures" / "Photos").mkdir(parents=True)
        (self.root / "Pictures" / "Photos" / "IMG_0042.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + b"\x00" * 50
        )
        (self.root / "Pictures" / "Vacation").mkdir(parents=True)
        (self.root / "Pictures" / "Vacation" / "beach_sunset.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + b"\x00" * 50
        )
        (self.root / "Pictures" / "family_photos.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        )

        # 2. Project source code files
        (self.root / "Projects" / "scoutx").mkdir(parents=True)
        (self.root / "Projects" / "scoutx" / "engine.py").write_text(
            "class ScoutXEngine:\n    def search(self): pass\n"
            + " ".join(f"engine_code_{i}" for i in range(80))
        )
        (self.root / "Projects" / "scoutx" / "launcher.py").write_text(
            "class DesktopLauncher:\n    def launch(self): pass\n"
            + " ".join(f"launcher_code_{i}" for i in range(80))
        )
        (self.root / "Projects" / "scoutx" / "main.py").write_text(
            "def main():\n    print('start')\n"
            + " ".join(f"main_code_{i}" for i in range(80))
        )

        # 3. Third-party package tree (simulating dependencies containing incidental keywords)
        (self.root / "Projects" / "yt_music_api").mkdir(parents=True)
        (self.root / "Projects" / "yt_music_api" / "yt_music_api.py").write_text(
            "# yt_music_api helper for album photo thumbnails\n"
            "def fetch_thumbnail_photo(): pass\n"
            + " ".join(f"music_code_{i}" for i in range(150))
        )
        (self.root / "Projects" / "yt_music_api" / "setup.py").write_text(
            "import setuptools\n" + " ".join(f"setup_code_{i}" for i in range(80))
        )

        self.engine = ScoutXEngine.from_directory(self.root)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_01_exact_filename_ranks_above_unrelated_candidates(self) -> None:
        """Exact filename or basename match ranks above unrelated items."""
        res = self.engine.search("main.py", limit=5)
        self.assertGreaterEqual(len(res.recommendations), 1)
        top = res.recommendations[0]
        self.assertEqual(top.document.path.name, "main.py")
        self.assertEqual(top.score, 1.0)
        self.assertFalse(res.router_result.forced_exit)
        self.assertTrue(res.router_result.trace[-1].reflection.confident)

    def test_02_photos_query_returns_strong_matching_item_near_top(self) -> None:
        """The query 'photos' ranks the Photos folder and photo files at the top."""
        res = self.engine.search("photos", limit=5)
        self.assertGreaterEqual(len(res.recommendations), 2)
        top_names = [rec.document.path.name for rec in res.recommendations[:3]]
        self.assertIn("Photos", top_names)
        self.assertIn("family_photos.png", top_names)
        # Verify unrelated yt_music_api is not ranked first
        self.assertNotEqual(res.recommendations[0].document.path.name, "yt_music_api.py")
        self.assertFalse(res.router_result.forced_exit)
        self.assertTrue(res.router_result.trace[-1].reflection.confident)

    def test_03_photo_singular_query_ranks_photo_items_without_guard_failure(self) -> None:
        """The singular query 'photo' expands to photos/images and terminates confidently."""
        res = self.engine.search("photo", limit=5)
        self.assertGreaterEqual(len(res.recommendations), 1)
        top_doc = res.recommendations[0].document
        self.assertIn(top_doc.path.name, ("Photos", "family_photos.png", "Pictures"))
        self.assertFalse(res.router_result.forced_exit)
        self.assertTrue(res.router_result.trace[-1].reflection.confident)

    def test_04_genuinely_unmatched_query_produces_no_falsely_confident_recommendations(self) -> None:
        """A nonsense query does not produce falsely confident recommendations."""
        res = self.engine.search("zzzxqv_unmatched_term", limit=5)
        self.assertEqual(len(res.recommendations), 0)
        self.assertTrue(res.router_result.forced_exit)
        self.assertFalse(res.router_result.trace[-1].reflection.confident)

    def test_05_media_files_are_indexed_by_filename_and_metadata_without_content_claim(self) -> None:
        """Binary media files (.jpg, .png) are indexed metadata-only and searchable by name."""
        res = self.engine.search("beach_sunset.jpg", limit=3)
        self.assertGreaterEqual(len(res.recommendations), 1)
        top = res.recommendations[0]
        self.assertEqual(top.document.path.name, "beach_sunset.jpg")
        self.assertEqual(top.document.content, "")  # No fake content extracted
        self.assertEqual(top.score, 1.0)

    def test_06_folder_and_bundle_exclusion_prunes_environment_noise(self) -> None:
        """Default directory exclusion rules skip venv and internal directories."""
        loader = DocumentLoader()
        with tempfile.TemporaryDirectory() as tmp:
            troot = Path(tmp)
            (troot / ".venv" / "lib" / "site-packages" / "pip").mkdir(parents=True)
            (troot / ".venv" / "lib" / "site-packages" / "pip" / "main.py").write_text("pass")
            (troot / "__pycache__").mkdir()
            (troot / "__pycache__" / "cached.pyc").write_bytes(b"pyc")
            (troot / "Photos Library.photoslibrary" / "database").mkdir(parents=True)
            (troot / "Photos Library.photoslibrary" / "database" / "internal.sqlite").write_text("sql")
            (troot / "Photos").mkdir()
            (troot / "Photos" / "sample.jpg").write_bytes(b"\xff\xd8\xff\xe0")

            docs = loader.load_directory(troot)
            doc_paths = [str(d.path.relative_to(troot)) for d in docs]

            # Internal implementation files are excluded
            self.assertFalse(any(".venv" in p for p in doc_paths))
            self.assertFalse(any("__pycache__" in p for p in doc_paths))
            self.assertFalse(any("internal.sqlite" in p for p in doc_paths))
            # Top-level photo folder and photoslibrary bundle are preserved
            self.assertTrue(any("Photos Library.photoslibrary" in p for p in doc_paths))
            self.assertTrue(any("sample.jpg" in p for p in doc_paths))

    def test_07_metadata_vector_does_not_leak_absolute_ancestor_paths(self) -> None:
        """Corpus metadata vector includes only basename, parent name, kind, and extension."""
        doc = FileDocument(
            id="doc1",
            path=Path("/Users/alice/Deep/Secret/Folder/report.pdf"),
            title="report",
            extension="pdf",
            content="",
            metadata={
                "name": "report.pdf",
                "extension": "pdf",
                "kind": "file",
                "parent": "/Users/alice/Deep/Secret/Folder",
                "path": "/Users/alice/Deep/Secret/Folder/report.pdf",
            },
        )
        corpus = DocumentCorpus((doc,))
        meta_vec = corpus.metadata_vector(doc)
        # Ancestor tokens must NOT be present in metadata vector
        self.assertNotIn("alice", meta_vec)
        self.assertNotIn("secret", meta_vec)
        # Immediate parent, basename, kind, extension are present
        self.assertIn("folder", meta_vec)
        self.assertIn("report", meta_vec)
        self.assertIn("pdf", meta_vec)

    def test_08_escalation_guard_terminates_under_persistent_uncertainty(self) -> None:
        """The bounded loop preserves the escalation guard under persistent uncertainty."""
        res = self.engine.search("that thing about numbers from a while ago", limit=5)
        rr = res.router_result
        self.assertEqual(len(rr.trace), 3)
        self.assertTrue(rr.forced_exit)
        self.assertFalse(rr.trace[-1].reflection.confident)
        self.assertIn("iteration guard forced Deep Lane exit", rr.trace[-1].reflection.reason)
