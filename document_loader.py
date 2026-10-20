"""Load files into ScoutX's document model with bounded, safe ingestion."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from xml.etree import ElementTree

from scoutx.models import FileDocument


PLAIN_TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".dart",
    ".go",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

BINARY_OR_UNSUPPORTED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".db",
    ".dmg",
    ".dll",
    ".dylib",
    ".exe",
    ".flac",
    ".gif",
    ".gz",
    ".ico",
    ".iso",
    ".jar",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".ogg",
    ".pdf",
    ".png",
    ".pyc",
    ".pyo",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".ttf",
    ".wasm",
    ".wav",
    ".webm",
    ".webp",
    ".whl",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
}

DEFAULT_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "dist",
        "build",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
    }
)
DEFAULT_EXCLUDED_DIRECTORY_SUFFIXES = (".photoslibrary", ".egg-info", ".app")

DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB plain-text file cap
DEFAULT_MAX_OFFICE_ARCHIVE_BYTES = 10 * 1024 * 1024  # 10 MiB compressed Office archive cap
DEFAULT_MAX_OFFICE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024  # 8 MiB cumulative XML member cap
DEFAULT_MAX_EXTRACTED_CHARS = 200_000  # 200k character cap per document
STREAM_CHUNK_BYTES = 65_536  # 64 KiB streaming read chunk
BINARY_SNIFF_BYTES = 4_096  # 4 KiB NUL-byte sniff window for unknown extensions


@dataclass(frozen=True)
class IngestionIssue:
    """Non-fatal ingestion diagnostic recorded for skipped, truncated, or unreadable files."""

    path: Path
    reason: str
    detail: str = ""


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer >= 1, got {value!r}")
    return value


class DocumentLoader:
    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_office_archive_bytes: int = DEFAULT_MAX_OFFICE_ARCHIVE_BYTES,
        max_office_uncompressed_bytes: int = DEFAULT_MAX_OFFICE_UNCOMPRESSED_BYTES,
        max_extracted_chars: int = DEFAULT_MAX_EXTRACTED_CHARS,
    ):
        self.max_file_bytes = _validate_positive_int("max_file_bytes", max_file_bytes)
        self.max_office_archive_bytes = _validate_positive_int(
            "max_office_archive_bytes", max_office_archive_bytes
        )
        self.max_office_uncompressed_bytes = _validate_positive_int(
            "max_office_uncompressed_bytes", max_office_uncompressed_bytes
        )
        self.max_extracted_chars = _validate_positive_int(
            "max_extracted_chars", max_extracted_chars
        )
        self._issues: list[IngestionIssue] = []

    @property
    def last_issues(self) -> tuple[IngestionIssue, ...]:
        """Return diagnostics collected during the most recent load operation."""
        return tuple(self._issues)

    def load_directory(self, root: Path) -> tuple[FileDocument, ...]:
        self._issues = []
        documents: list[FileDocument] = []
        root_path = Path(root)
        try:
            entries = sorted(root_path.rglob("*"))
        except OSError as exc:
            self._record_issue(root_path, "unreadable", f"Failed to scan directory: {exc}")
            return ()

        for path in entries:
            if self._is_hidden(path, root_path):
                continue
            document = self._load_path_internal(path)
            if document is not None:
                documents.append(document)
        return tuple(documents)

    def load_directory_with_issues(
        self, root: Path
    ) -> tuple[tuple[FileDocument, ...], tuple[IngestionIssue, ...]]:
        documents = self.load_directory(root)
        return documents, self.last_issues

    def load_path(self, path: Path) -> FileDocument | None:
        self._issues = []
        return self._load_path_internal(Path(path))

    def _load_path_internal(self, path: Path) -> FileDocument | None:
        try:
            if path.is_dir():
                return self._load_folder_internal(path)
            if path.is_file():
                return self._load_file_internal(path)
        except OSError as exc:
            self._record_issue(path, "unreadable", f"Filesystem access error: {exc}")
            return None
        self._record_issue(path, "unreadable", "Path is neither a regular file nor a directory")
        return None

    def load_folder(self, path: Path) -> FileDocument | None:
        self._issues = []
        return self._load_folder_internal(Path(path))

    def _load_folder_internal(self, path: Path) -> FileDocument | None:
        try:
            stat = path.stat()
        except OSError as exc:
            self._record_issue(path, "unreadable", f"Cannot stat folder: {exc}")
            return None

        child_names: list[str] = []
        try:
            child_names = [
                child.name
                for child in sorted(path.iterdir())[:100]
                if not child.name.startswith(".")
            ]
        except OSError as exc:
            self._record_issue(path, "unreadable", f"Cannot list folder contents: {exc}")
            child_names = []

        content = self._truncate_content(path, " ".join(child_names))
        return FileDocument(
            id=self._document_id(path),
            path=path,
            title=path.name,
            extension="folder",
            content=content,
            is_directory=True,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            metadata={
                "name": path.name,
                "extension": "folder",
                "kind": "directory",
                "parent": str(path.parent),
                "path": str(path),
            },
        )

    def load_file(self, path: Path) -> FileDocument | None:
        self._issues = []
        return self._load_file_internal(Path(path))

    def _load_file_internal(self, path: Path) -> FileDocument | None:
        try:
            if path.is_dir():
                return self._load_folder_internal(path)
            stat = path.stat()
        except OSError as exc:
            self._record_issue(path, "unreadable", f"Cannot stat file: {exc}")
            return None

        suffix = path.suffix.lower()
        content = ""

        if suffix in OFFICE_EXTENSIONS:
            if stat.st_size > self.max_office_archive_bytes:
                self._record_issue(
                    path,
                    "office_archive_too_large",
                    f"Office archive size {stat.st_size} exceeds limit {self.max_office_archive_bytes}",
                )
            else:
                content = self._read_office_text(path)
        elif suffix in PLAIN_TEXT_EXTENSIONS:
            if stat.st_size > self.max_file_bytes:
                self._record_issue(
                    path,
                    "file_too_large",
                    f"File size {stat.st_size} exceeds limit {self.max_file_bytes}",
                )
            else:
                content = self._read_plain_text(path, check_binary=False)
        elif suffix in BINARY_OR_UNSUPPORTED_EXTENSIONS:
            self._record_issue(
                path,
                "binary_or_unsupported",
                f"Extension '{suffix}' is treated as metadata-only binary/unsupported",
            )
            content = ""
        else:
            if stat.st_size > self.max_file_bytes:
                self._record_issue(
                    path,
                    "file_too_large",
                    f"File size {stat.st_size} exceeds limit {self.max_file_bytes}",
                )
            else:
                content = self._read_plain_text(path, check_binary=True)

        content = self._truncate_content(path, content)

        return FileDocument(
            id=self._document_id(path),
            path=path,
            title=path.stem,
            extension=suffix.removeprefix("."),
            content=content,
            is_directory=False,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            created_at=datetime.fromtimestamp(stat.st_ctime),
            metadata={
                "name": path.name,
                "extension": suffix.removeprefix("."),
                "kind": "file",
                "parent": str(path.parent),
                "path": str(path),
            },
        )

    def _read_plain_text(self, path: Path, *, check_binary: bool = False) -> str:
        try:
            with path.open("rb") as handle:
                if check_binary:
                    sample = handle.read(min(BINARY_SNIFF_BYTES, self.max_file_bytes + 1))
                    if b"\x00" in sample:
                        self._record_issue(
                            path,
                            "binary_or_unsupported",
                            "Detected NUL byte in file header; skipping text extraction",
                        )
                        return ""
                    remaining = self._read_stream_bounded(
                        handle, max_bytes=max(0, self.max_file_bytes - len(sample))
                    )
                    if remaining is None:
                        self._record_issue(
                            path,
                            "file_too_large",
                            f"File stream exceeded limit {self.max_file_bytes} bytes while reading",
                        )
                        return ""
                    raw_bytes = sample + remaining
                else:
                    raw_bytes = self._read_stream_bounded(handle, max_bytes=self.max_file_bytes)
                    if raw_bytes is None:
                        self._record_issue(
                            path,
                            "file_too_large",
                            f"File stream exceeded limit {self.max_file_bytes} bytes while reading",
                        )
                        return ""
            return raw_bytes.decode("utf-8", errors="ignore")
        except OSError as exc:
            self._record_issue(path, "unreadable", f"Failed to read plain text file: {exc}")
            return ""

    def _read_office_text(self, path: Path) -> str:
        try:
            with zipfile.ZipFile(path) as archive:
                return self._extract_office_archive_text(archive, path)
        except (
            OSError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
            KeyError,
            ElementTree.ParseError,
        ) as exc:
            self._record_issue(
                path,
                "corrupt_office_archive",
                f"Failed to extract Office XML archive: {exc}",
            )
            return ""

    def _extract_office_archive_text(self, archive: zipfile.ZipFile, path: Path) -> str:
        names = self._office_text_members(path.suffix.lower(), archive.namelist())
        chunks: list[str] = []
        remaining_budget = self.max_office_uncompressed_bytes

        for name in names:
            info = archive.getinfo(name)
            if info.file_size > remaining_budget:
                self._record_issue(
                    path,
                    "office_uncompressed_too_large",
                    f"Declared XML member '{name}' size ({info.file_size}) exceeds remaining uncompressed budget ({remaining_budget})",
                )
                return ""
            with archive.open(name) as member:
                member_bytes = self._read_stream_bounded(member, max_bytes=remaining_budget)
            if member_bytes is None:
                self._record_issue(
                    path,
                    "office_uncompressed_too_large",
                    f"Decompressed stream for XML member '{name}' exceeded cumulative budget ({self.max_office_uncompressed_bytes} bytes)",
                )
                return ""
            remaining_budget -= len(member_bytes)
            xml = ElementTree.fromstring(member_bytes)
            chunks.extend(text.strip() for text in xml.itertext() if text.strip())

        return " ".join(chunks)

    @staticmethod
    def _read_stream_bounded(stream: BinaryIO, *, max_bytes: int) -> bytes | None:
        """Read from a binary stream in fixed chunks, returning None if stream exceeds max_bytes."""
        parts: list[bytes] = []
        total = 0
        while True:
            read_size = min(STREAM_CHUNK_BYTES, max_bytes - total + 1)
            chunk = stream.read(read_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                return None
            parts.append(chunk)
        return b"".join(parts)

    def _truncate_content(self, path: Path, content: str) -> str:
        if len(content) > self.max_extracted_chars:
            self._record_issue(
                path,
                "content_truncated",
                f"Extracted content length {len(content)} truncated to {self.max_extracted_chars} chars",
            )
            return content[: self.max_extracted_chars]
        return content

    def _record_issue(self, path: Path, reason: str, detail: str = "") -> None:
        self._issues.append(IngestionIssue(path=path, reason=reason, detail=detail))

    def _office_text_members(self, suffix: str, names: list[str]) -> list[str]:
        if suffix == ".docx":
            return [name for name in names if name == "word/document.xml"]
        if suffix == ".pptx":
            return sorted(
                name
                for name in names
                if name.startswith("ppt/slides/") and name.endswith(".xml")
            )
        if suffix == ".xlsx":
            return [name for name in names if name in {"xl/sharedStrings.xml", "xl/workbook.xml"}]
        return []

    def _document_id(self, path: Path) -> str:
        return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]

    def _is_hidden(self, path: Path, root: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        for idx, part in enumerate(relative.parts):
            if part.startswith("."):
                return True
            if part in DEFAULT_EXCLUDED_DIRECTORY_NAMES:
                return True
            # For bundle directories (e.g. *.photoslibrary), index the top-level bundle
            # directory itself, but skip its internal implementation details.
            if any(part.endswith(sfx) for sfx in DEFAULT_EXCLUDED_DIRECTORY_SUFFIXES) and idx < len(relative.parts) - 1:
                return True
        return False
