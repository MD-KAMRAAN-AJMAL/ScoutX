"""Docling document parsing with an on-demand Apache Tika fallback."""

from __future__ import annotations

import csv
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

from app.ingestion.normalized import ExtractedBlock, ExtractedDocument, ExtractionStatus, make_block
from app.security.parser_sandbox import ParserInputError, validate_input

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}


class DoclingParser:
    name = "docling"

    def parse(self, path: str | Path, file_id: UUID, content_version: int) -> ExtractedDocument:
        path = Path(path)
        try:
            path = validate_input(path)
        except (OSError, ParserInputError) as exc:
            return self._failed(file_id, content_version, str(exc))
        if path.suffix.lower() not in DOCUMENT_EXTENSIONS:
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.UNSUPPORTED,
                blocks=[],
                parser_name=self.name,
                parser_version="unknown",
                warnings=["Docling parser does not handle this extension"],
            )
        warnings: list[str] = []
        result = None
        try:
            from docling.document_converter import DocumentConverter

            result = DocumentConverter().convert(str(path))
            parser_version = version("docling")
            document = result.document
            blocks = self._extract_items(
                document,
                file_id,
                content_version,
                parser_version,
                is_presentation=path.suffix.lower() == ".pptx",
            )
            if not blocks:
                markdown = document.export_to_markdown()
                if markdown and markdown.strip():
                    blocks = [
                        make_block(
                            markdown.strip(),
                            file_id=file_id,
                            content_version=content_version,
                            parser_name=self.name,
                            parser_version=parser_version,
                            content_type="document",
                            metadata={"source": "markdown_export"},
                        )
                    ]
            if blocks:
                return ExtractedDocument(
                    file_id=file_id,
                    content_version=content_version,
                    status=ExtractionStatus.SUCCESS,
                    blocks=blocks,
                    parser_name=self.name,
                    parser_version=parser_version,
                )
            warnings.append("Docling completed without extracting searchable content")
        except Exception as exc:  # noqa: BLE001 - isolate parser/model failures to this file
            warnings.append(f"Docling failed: {type(exc).__name__}: {exc}")
            parser_version = "unavailable"
        finally:
            try:
                source = getattr(result, "input", None)
                release = getattr(source, "release", None)
                if callable(release):
                    release()
            except Exception as exc:  # noqa: BLE001 - a release failure must not replace extraction output
                warnings.append(f"Docling input cleanup failed: {type(exc).__name__}: {exc}")
        fallback = self._tika(path, file_id, content_version, warnings)
        if fallback is not None:
            return fallback
        return ExtractedDocument(
            file_id=file_id,
            content_version=content_version,
            status=ExtractionStatus.FAILED if warnings else ExtractionStatus.PARTIAL,
            blocks=[],
            parser_name=self.name,
            parser_version=parser_version,
            warnings=warnings,
            error="No document parser produced extractable content",
        )

    def _extract_items(
        self,
        document,
        file_id: UUID,
        content_version: int,
        parser_version: str,
        *,
        is_presentation: bool = False,
    ) -> list[ExtractedBlock]:
        blocks: list[ExtractedBlock] = []
        iterator = getattr(document, "iterate_items", None)
        if not callable(iterator):
            return blocks
        for item, _level in iterator():
            text = getattr(item, "text", None)
            label = str(getattr(item, "label", "text"))
            if not text and hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(document=document)
                except Exception:  # noqa: BLE001 - table export fallback is best effort
                    text = None
            if not text or not str(text).strip():
                continue
            provenance = getattr(item, "prov", None) or []
            page = getattr(provenance[0], "page_no", None) if provenance else None
            page = int(page) if page is not None else None
            title = getattr(item, "section_header", None) or getattr(item, "name", None)
            block_type = (
                "table"
                if "table" in label.lower()
                else "heading"
                if "heading" in label.lower() or "title" in label.lower()
                else "list"
                if "list" in label.lower()
                else "text"
            )
            blocks.append(
                make_block(
                    str(text).strip(),
                    file_id=file_id,
                    content_version=content_version,
                    parser_name=self.name,
                    parser_version=parser_version,
                    content_type=block_type,
                    page_number=None if is_presentation else page,
                    slide_number=page if is_presentation else None,
                    section_title=str(title) if title else None,
                    metadata={"docling_label": label},
                )
            )
        return blocks

    @staticmethod
    def _tika(
        path: Path, file_id: UUID, content_version: int, prior_warnings: list[str]
    ) -> ExtractedDocument | None:
        try:
            from tika import parser

            parsed = parser.from_file(str(path), xmlContent=False, requestOptions={"timeout": 90})
            content = (parsed or {}).get("content")
            if not content or not content.strip():
                prior_warnings.append("Apache Tika fallback returned no text")
                return None
            tika_version = version("tika")
            block = make_block(
                content.strip(),
                file_id=file_id,
                content_version=content_version,
                parser_name="apache-tika",
                parser_version=tika_version,
                content_type="document",
                metadata={"metadata": (parsed or {}).get("metadata", {})},
            )
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.PARTIAL,
                blocks=[block],
                parser_name="apache-tika",
                parser_version=tika_version,
                warnings=[
                    *prior_warnings,
                    "Used Tika fallback; structural provenance may be unavailable",
                ],
            )
        except Exception as exc:  # noqa: BLE001 - Tika errors become extraction warnings
            prior_warnings.append(f"Apache Tika fallback failed: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _failed(file_id, content_version, error):
        return ExtractedDocument(
            file_id=file_id,
            content_version=content_version,
            status=ExtractionStatus.FAILED,
            blocks=[],
            parser_name="docling",
            parser_version="unknown",
            error=error,
        )


class SpreadsheetParser:
    """CSV/XLSX reader preserving sheet and cell-range provenance."""

    name = "spreadsheet"

    def parse(self, path: str | Path, file_id: UUID, content_version: int) -> ExtractedDocument:
        path = Path(path)
        try:
            path = validate_input(path)
            if path.stat().st_size == 0:
                raise ValueError("Spreadsheet is empty")
            blocks = (
                self._xlsx(path, file_id, content_version)
                if path.suffix.lower() == ".xlsx"
                else self._csv(path, file_id, content_version)
            )
        except Exception as exc:  # noqa: BLE001 - parser errors are returned as per-file failures
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.FAILED,
                blocks=[],
                parser_name=self.name,
                parser_version="1",
                error=f"{type(exc).__name__}: {exc}",
            )
        if not blocks:
            return ExtractedDocument(
                file_id=file_id,
                content_version=content_version,
                status=ExtractionStatus.PARTIAL,
                blocks=[],
                parser_name=self.name,
                parser_version="1",
                warnings=["No non-empty spreadsheet rows found"],
            )
        return ExtractedDocument(
            file_id=file_id,
            content_version=content_version,
            status=ExtractionStatus.SUCCESS,
            blocks=blocks,
            parser_name=self.name,
            parser_version="1",
        )

    def _xlsx(self, path, file_id, content_version):
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        blocks = []
        try:
            for sheet in workbook.worksheets:
                rows = sheet.iter_rows(values_only=True)
                header = next(rows, None)
                if not header:
                    continue
                headers = [
                    str(value).strip() if value is not None else f"column_{i + 1}"
                    for i, value in enumerate(header)
                ]
                nonempty_header = any(value is not None for value in header)
                if nonempty_header:
                    blocks.append(
                        make_block(
                            " | ".join(headers),
                            file_id=file_id,
                            content_version=content_version,
                            parser_name=self.name,
                            parser_version="openpyxl",
                            content_type="spreadsheet_header",
                            sheet_name=sheet.title,
                            cell_range=f"A1:{sheet.cell(1, max(1, len(headers))).column_letter}1",
                        )
                    )
                group, start_row = [], 2
                for row_no, values in enumerate(rows, start=2):
                    if not any(value is not None for value in values):
                        if group:
                            blocks.append(
                                self._rows_block(
                                    group,
                                    headers,
                                    sheet.title,
                                    start_row,
                                    row_no - 1,
                                    file_id,
                                    content_version,
                                )
                            )
                            group = []
                        start_row = row_no + 1
                        continue
                    group.append((row_no, values))
                    if len(group) >= 25:
                        blocks.append(
                            self._rows_block(
                                group,
                                headers,
                                sheet.title,
                                start_row,
                                row_no,
                                file_id,
                                content_version,
                            )
                        )
                        group = []
                        start_row = row_no + 1
                if group:
                    blocks.append(
                        self._rows_block(
                            group,
                            headers,
                            sheet.title,
                            start_row,
                            group[-1][0],
                            file_id,
                            content_version,
                        )
                    )
        finally:
            workbook.close()
        return blocks

    def _csv(self, path, file_id, content_version):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows:
            return []
        headers = [value.strip() or f"column_{i + 1}" for i, value in enumerate(rows[0])]
        blocks = [
            make_block(
                " | ".join(headers),
                file_id=file_id,
                content_version=content_version,
                parser_name=self.name,
                parser_version="stdlib-csv",
                content_type="spreadsheet_header",
                sheet_name="CSV",
                cell_range=f"A1:{_column_name(len(headers))}1",
            )
        ]
        for start in range(1, len(rows), 25):
            group = [
                (row_no + 1, row)
                for row_no, row in enumerate(rows[start : start + 25], start=start)
            ]
            blocks.append(
                self._rows_block(
                    group, headers, "CSV", group[0][0], group[-1][0], file_id, content_version
                )
            )
        return blocks

    def _rows_block(self, rows, headers, sheet, start, end, file_id, content_version):
        content = "\n".join(
            " | ".join(
                f"{headers[i] if i < len(headers) else f'column_{i + 1}'}: {value if value is not None else ''}"
                for i, value in enumerate(values)
                if value is not None
            )
            for _row, values in rows
        )
        max_columns = max((len(values) for _row, values in rows), default=1)
        return make_block(
            content,
            file_id=file_id,
            content_version=content_version,
            parser_name=self.name,
            parser_version="openpyxl-or-csv",
            content_type="spreadsheet_rows",
            sheet_name=sheet,
            cell_range=f"A{start}:{_column_name(max_columns)}{end}",
            metadata={"row_numbers": [row for row, _ in rows]},
        )


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result or "A"
