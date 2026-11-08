"""Format-based parser selection for content extraction."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID

from app.ingestion.code_parser import LANGUAGE_BY_SUFFIX, CodeParser
from app.ingestion.docling_parser import DOCUMENT_EXTENSIONS, DoclingParser, SpreadsheetParser
from app.ingestion.image_parser import IMAGE_EXTENSIONS, ImageParser
from app.ingestion.media_parser import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, MediaParser
from app.ingestion.normalized import ExtractedDocument, ExtractionStatus, make_block
from app.security.parser_sandbox import DEFAULT_LIMITS, validate_input

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".htm",
}
SPREADSHEET_EXTENSIONS = {".xlsx", ".csv"}
PARSER_VERSION = "1.0"


class PlainTextParser:
    name = "plain-text"

    def parse(self, path: Path, file_id: UUID, content_version: int) -> ExtractedDocument:
        try:
            source = validate_input(path, max_bytes=DEFAULT_LIMITS.max_text_bytes)
            text = source.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                raise ValueError("Text file is empty")
            warnings = ["Invalid UTF-8 bytes were replaced"] if "\ufffd" in text else []
            block = make_block(
                text,
                file_id=file_id,
                content_version=content_version,
                parser_name=self.name,
                parser_version=PARSER_VERSION,
                content_type="text",
                line_start=1,
                line_end=max(1, len(text.splitlines())),
                section_title=source.name,
            )
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.PARTIAL if warnings else ExtractionStatus.SUCCESS,
                blocks=[block],
                parser_name=self.name,
                parser_version=PARSER_VERSION,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 - return parser failures in the normalized result
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.FAILED,
                blocks=[],
                parser_name=self.name,
                parser_version=PARSER_VERSION,
                error=f"{type(exc).__name__}: {exc}",
            )


class ParserRegistry:
    """Routes each file to one appropriate parser, with errors isolated per file."""

    def __init__(
        self,
        *,
        code_parser=None,
        docling_parser=None,
        image_parser=None,
        media_parser=None,
        spreadsheet_parser=None,
        text_parser=None,
    ):
        self.code_parser = code_parser or CodeParser()
        self.docling_parser = docling_parser or DoclingParser()
        self.image_parser = image_parser or ImageParser()
        self.media_parser = media_parser or MediaParser()
        self.spreadsheet_parser = spreadsheet_parser or SpreadsheetParser()
        self.text_parser = text_parser or PlainTextParser()

    def select_parser(self, path: str | Path, mime_type: str | None = None):
        """Return a single parser based on trusted extension and MIME hints."""
        path = Path(path)
        ext = path.suffix.lower()
        mime = (mime_type or "").lower().split(";", 1)[0]
        if ext in LANGUAGE_BY_SUFFIX or mime.startswith(
            (
                "text/x-python",
                "text/x-c",
                "text/x-java",
                "text/javascript",
                "application/javascript",
            )
        ):
            return self.code_parser
        if ext in DOCUMENT_EXTENSIONS or mime in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }:
            return self.docling_parser
        if ext in SPREADSHEET_EXTENSIONS or mime in {
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }:
            return self.spreadsheet_parser
        if ext in IMAGE_EXTENSIONS or mime.startswith("image/"):
            return self.image_parser
        if ext in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS or mime.startswith(("audio/", "video/")):
            return self.media_parser
        if ext in TEXT_EXTENSIONS or mime.startswith("text/"):
            return self.text_parser
        return None

    def parse(
        self, path: str | Path, file_id: UUID, content_version: int, mime_type: str | None = None
    ) -> ExtractedDocument:
        """Parse one source file and always return the normalized result schema."""
        path = Path(path)
        parser = self.select_parser(path, mime_type)
        if parser is None:
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.UNSUPPORTED,
                blocks=[],
                parser_name="none",
                parser_version="1",
                warnings=[
                    f"No parser registered for {path.suffix or mime_type or 'unknown format'}"
                ],
            )
        try:
            result = parser.parse(path, file_id, content_version)
            if result.file_id != file_id or result.content_version != content_version:
                raise ValueError("Parser returned mismatched file identity or content version")
            return result
        except Exception as exc:  # noqa: BLE001 - a parser failure must not fail the indexing batch
            # A broken parser/model for one file must not fail a Celery batch.
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.FAILED,
                blocks=[],
                parser_name=getattr(parser, "name", type(parser).__name__),
                parser_version=_parser_version(parser),
                error=f"{type(exc).__name__}: {exc}",
            )


@lru_cache(maxsize=1)
def get_default_registry() -> ParserRegistry:
    """Reuse lazy parser instances within a worker process (not across processes)."""
    return ParserRegistry()


def _parser_version(parser) -> str:
    package_by_parser = {
        "docling": "docling",
        "tree-sitter": "tree-sitter-language-pack",
        "opencv-paddleocr": "paddleocr",
        "ffmpeg-faster-whisper": "faster-whisper",
    }
    package = package_by_parser.get(getattr(parser, "name", ""))
    if package:
        try:
            return version(package)
        except PackageNotFoundError:
            pass
    return "unknown"
