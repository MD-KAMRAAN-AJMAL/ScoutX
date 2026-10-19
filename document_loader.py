"""Load files into ScoutX's document model."""

from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime
from pathlib import Path
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


class DocumentLoader:
    def load_directory(self, root: Path) -> tuple[FileDocument, ...]:
        documents: list[FileDocument] = []
        for path in sorted(root.rglob("*")):
            if self._is_hidden(path, root):
                continue
            document = self.load_path(path)
            if document is not None:
                documents.append(document)
        return tuple(documents)

    def load_path(self, path: Path) -> FileDocument | None:
        if path.is_dir():
            return self.load_folder(path)
        if path.is_file():
            return self.load_file(path)
        return None

    def load_folder(self, path: Path) -> FileDocument | None:
        try:
            stat = path.stat()
        except OSError:
            return None

        child_names: list[str] = []
        try:
            child_names = [child.name for child in sorted(path.iterdir())[:100] if not child.name.startswith(".")]
        except OSError:
            child_names = []

        return FileDocument(
            id=self._document_id(path),
            path=path,
            title=path.name,
            extension="folder",
            content=" ".join(child_names),
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
        if path.is_dir():
            return self.load_folder(path)
        suffix = path.suffix.lower()
        try:
            stat = path.stat()
        except OSError:
            return None

        content = ""
        if suffix in OFFICE_EXTENSIONS:
            content = self._read_office_text(path) or self._read_plain_text(path)
        elif suffix in PLAIN_TEXT_EXTENSIONS:
            content = self._read_plain_text(path)
        elif suffix == ".pdf":
            content = ""
        else:
            content = self._read_plain_text(path)

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

    def _read_plain_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def _read_office_text(self, path: Path) -> str:
        try:
            with zipfile.ZipFile(path) as archive:
                names = self._office_text_members(path.suffix.lower(), archive.namelist())
                chunks: list[str] = []
                for name in names:
                    with archive.open(name) as member:
                        xml = ElementTree.fromstring(member.read())
                    chunks.extend(text.strip() for text in xml.itertext() if text.strip())
                return " ".join(chunks)
        except (OSError, zipfile.BadZipFile, ElementTree.ParseError):
            return ""

    def _office_text_members(self, suffix: str, names: list[str]) -> list[str]:
        if suffix == ".docx":
            return [name for name in names if name == "word/document.xml"]
        if suffix == ".pptx":
            return [name for name in names if name.startswith("ppt/slides/") and name.endswith(".xml")]
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
        return any(part.startswith(".") for part in relative.parts)
