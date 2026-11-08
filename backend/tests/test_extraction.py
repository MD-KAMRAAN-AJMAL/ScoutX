from __future__ import annotations

import sys
import types
import uuid
import zipfile

import pytest

from app.ingestion.code_parser import CodeParser
from app.ingestion.docling_parser import DoclingParser, SpreadsheetParser
from app.ingestion.normalized import ExtractedBlock, ExtractionStatus
from app.ingestion.parser_registry import ParserRegistry

FILE_ID = uuid.uuid4()


def test_normalized_block_requires_identity_and_ranges():
    block = ExtractedBlock(content="hello", file_id=FILE_ID, content_version=1,
                           parser_name="test", parser_version="1", line_start=1, line_end=2)
    assert block.content == "hello"
    with pytest.raises(ValueError):
        ExtractedBlock(content="bad", file_id=FILE_ID, content_version=1,
                       parser_name="test", parser_version="1", line_start=2, line_end=1)


def test_registry_routes_one_parser_and_reports_unsupported(tmp_path):
    file = tmp_path / "unknown.bin"
    file.write_bytes(b"\x00\x01")
    registry = ParserRegistry()
    assert registry.select_parser(file) is None
    result = registry.parse(file, FILE_ID, 1)
    assert result.status is ExtractionStatus.UNSUPPORTED
    assert result.blocks == []


def test_code_plain_text_fallback_keeps_line_provenance(tmp_path):
    file = tmp_path / "sample.py"
    file.write_text("def answer():\n    return 42\n", encoding="utf-8")
    result = CodeParser().parse(file, FILE_ID, 4)
    assert result.status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
    assert result.blocks
    assert all(block.file_id == FILE_ID and block.content_version == 4 for block in result.blocks)
    assert all(block.line_start and block.line_end for block in result.blocks)


def test_csv_has_headers_and_ranges(tmp_path):
    file = tmp_path / "sales.csv"
    file.write_text("name,total\nAda,3\nGrace,5\n", encoding="utf-8")
    result = SpreadsheetParser().parse(file, FILE_ID, 2)
    assert result.status is ExtractionStatus.SUCCESS
    assert result.blocks[0].content_type == "spreadsheet_header"
    assert result.blocks[0].sheet_name == "CSV"
    assert result.blocks[1].cell_range == "A2:B3"
    assert "Ada" in result.blocks[1].content


def test_empty_corrupt_and_unsupported_are_isolated(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.touch()
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"not a workbook")
    unsupported = tmp_path / "anything.xyz"
    unsupported.write_bytes(b"data")
    assert ParserRegistry().parse(empty, FILE_ID, 1).status is ExtractionStatus.FAILED
    assert SpreadsheetParser().parse(corrupt, FILE_ID, 1).status is ExtractionStatus.FAILED
    assert ParserRegistry().parse(unsupported, FILE_ID, 1).status is ExtractionStatus.UNSUPPORTED


def test_docling_uses_tika_only_after_primary_failure(tmp_path, monkeypatch):
    file = tmp_path / "document.pdf"
    file.write_bytes(b"%PDF-1.4\nminimal fixture")
    docling = types.ModuleType("docling")
    converter_module = types.ModuleType("docling.document_converter")

    class BrokenConverter:
        def convert(self, _path):
            raise RuntimeError("primary parser unavailable")

    converter_module.DocumentConverter = BrokenConverter
    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)
    tika = types.ModuleType("tika")
    tika.parser = types.SimpleNamespace(from_file=lambda *_a, **_k: {"content": "fallback text", "metadata": {}})
    monkeypatch.setitem(sys.modules, "tika", tika)
    monkeypatch.setattr("app.ingestion.docling_parser.version", lambda _name: "test-version")
    result = DoclingParser().parse(file, FILE_ID, 1)
    assert result.status is ExtractionStatus.PARTIAL
    assert result.parser_name == "apache-tika"
    assert result.blocks[0].content == "fallback text"


def test_docling_keeps_presentation_slide_number(tmp_path, monkeypatch):
    # A minimal OOXML package with one slide; the converter is isolated so this
    # test does not download Docling's layout/OCR models.
    file = tmp_path / "deck.pptx"
    with zipfile.ZipFile(file, "w") as package:
        package.writestr("[Content_Types].xml", "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'/>")
        package.writestr("ppt/presentation.xml", "<p:presentation xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main'/>")
        package.writestr("ppt/slides/slide1.xml", "<p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main'/>")

    class Provenance:
        page_no = 3

    class Item:
        text = "Slide content"
        label = "text"
        prov = (Provenance(),)

    class Document:
        def iterate_items(self):
            return [(Item(), 0)]

    class Converter:
        def convert(self, _path):
            return types.SimpleNamespace(document=Document())

    converter_module = types.ModuleType("docling.document_converter")
    converter_module.DocumentConverter = Converter
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)
    monkeypatch.setattr("app.ingestion.docling_parser.version", lambda _name: "fake")
    result = DoclingParser().parse(file, FILE_ID, 2)
    assert result.status is ExtractionStatus.SUCCESS
    assert result.blocks[0].slide_number == 3
    assert result.blocks[0].page_number is None


def test_xlsx_real_workbook_preserves_sheet_ranges(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    file = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Budget"
    sheet.append(["item", "amount"])
    sheet.append(["rent", 1200])
    workbook.save(file)
    result = SpreadsheetParser().parse(file, FILE_ID, 3)
    assert result.status is ExtractionStatus.SUCCESS
    assert result.blocks[1].sheet_name == "Budget"
    assert result.blocks[1].cell_range == "A2:B2"


def test_image_parser_with_real_png_and_fake_ocr(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    from app.ingestion.image_parser import ImageParser

    file = tmp_path / "sample.png"
    assert cv2.imwrite(str(file), numpy.full((32, 64, 3), 255, dtype=numpy.uint8))

    class FakeOcr:
        def predict(self, input):
            return [{"rec_texts": ["Invoice 123"], "rec_scores": [0.98], "dt_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]]}]

    parser = ImageParser(ocr_factory=FakeOcr)
    monkeypatch.setattr("app.ingestion.image_parser.version", lambda _name: "fake")
    result = parser.parse(file, FILE_ID, 1)
    assert result.status is ExtractionStatus.SUCCESS
    assert result.blocks[0].content == "Invoice 123"
    assert result.blocks[0].metadata["confidence"] == 0.98


def test_media_transcription_with_real_audio_fixture_and_fake_whisper(tmp_path, monkeypatch):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is not installed")
    from app.ingestion.media_parser import MediaParser

    file = tmp_path / "tone.wav"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(file)], check=True)

    class Segment:
        start, end, text = 0.1, 0.8, "test speech"

    class FakeModel:
        def transcribe(self, _path, **_kwargs):
            return iter([Segment()]), types.SimpleNamespace(language="en")

    parser = MediaParser(model_factory=FakeModel)
    monkeypatch.setattr("app.ingestion.media_parser.version", lambda _name: "fake")
    result = parser.parse(file, FILE_ID, 1)
    assert result.status is ExtractionStatus.SUCCESS
    assert result.blocks[0].start_time_ms == 100
    assert result.blocks[0].end_time_ms == 800
    assert result.blocks[0].language == "en"
