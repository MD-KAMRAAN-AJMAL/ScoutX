"""Tree-sitter based structural source-code extraction."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from uuid import UUID

from app.ingestion.normalized import ExtractedBlock, ExtractedDocument, ExtractionStatus, make_block
from app.security.parser_sandbox import DEFAULT_LIMITS, ParserInputError, validate_input

LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".java": "java",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".rs": "rust", ".go": "go",
}

_DECLARATION_TYPES = {
    "function_definition", "function_declaration", "method_definition", "method_declaration",
    "class_definition", "class_declaration", "class_specifier", "struct_item", "enum_item",
    "impl_item", "trait_item", "function_item", "interface_declaration", "type_declaration",
    "struct_declaration", "enum_declaration", "namespace_definition", "module", "method",
}
_IMPORT_TYPES = {"import_statement", "import_declaration", "import_from_statement", "use_declaration", "preproc_include"}
_COMMENT_TYPES = {"comment", "line_comment", "block_comment", "documentation_comment"}
_IDENTIFIER_TYPES = {"identifier", "type_identifier", "field_identifier", "property_identifier", "namespace_identifier"}


class CodeParser:
    name = "tree-sitter"

    def parse(self, path: str | Path, file_id: UUID, content_version: int) -> ExtractedDocument:
        file_path = Path(path)
        language = LANGUAGE_BY_SUFFIX.get(file_path.suffix.lower())
        if language is None:
            return ExtractedDocument(file_id=file_id, content_version=content_version, status=ExtractionStatus.UNSUPPORTED,
                                     blocks=[], parser_name=self.name, parser_version="unavailable", warnings=["Unsupported source extension"])
        try:
            source_path = validate_input(file_path, max_bytes=DEFAULT_LIMITS.max_text_bytes)
            raw = source_path.read_bytes()
        except (OSError, ParserInputError) as exc:
            return self._failed(file_id, content_version, str(exc))
        if not raw:
            return self._failed(file_id, content_version, "Source file is empty")
        source = raw.decode("utf-8", errors="replace")
        warnings: list[str] = []
        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(language)
            tree = parser.parse(raw)
            parser_version = version("tree-sitter-language-pack")
            blocks = self._blocks(tree.root_node, raw, file_id, content_version, language, parser_version)
            if tree.root_node.has_error:
                warnings.append("Source contains syntax errors; extracted nodes may be incomplete")
        except Exception as exc:
            parser_version = "unavailable"
            warnings.append(f"Tree-sitter grammar unavailable; returned plain-text source: {exc}")
            blocks = []
        if not blocks:
            blocks = [make_block(source, file_id=file_id, content_version=content_version, parser_name=self.name,
                                 parser_version=parser_version, content_type="source", language=language,
                                 line_start=1, line_end=max(1, len(source.splitlines())), section_title=file_path.name)]
        return ExtractedDocument(file_id=file_id, content_version=content_version,
                                 status=ExtractionStatus.PARTIAL if warnings else ExtractionStatus.SUCCESS,
                                 blocks=blocks, parser_name=self.name, parser_version=parser_version, warnings=warnings)

    def _blocks(self, root, raw: bytes, file_id: UUID, content_version: int, language: str, parser_version: str) -> list[ExtractedBlock]:
        blocks: list[ExtractedBlock] = []
        seen: set[tuple[str, int, int]] = set()

        def visit(node, parents: tuple[str, ...]) -> None:
            kind = node.type
            text = raw[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            line_start, line_end = node.start_point.row + 1, node.end_point.row + (1 if node.end_point.column else 0)
            if kind in _DECLARATION_TYPES:
                name_node = node.child_by_field_name("name")
                symbol_name = raw[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace") if name_node else kind
                section = ".".join((*parents, symbol_name))
                self._append(blocks, seen, text, "code_declaration", file_id, content_version, parser_version,
                             language, line_start, line_end, section, {"symbol_kind": kind, "symbol_name": symbol_name})
                for child in node.children:
                    visit(child, (*parents, symbol_name))
                return
            if kind in _IMPORT_TYPES:
                self._append(blocks, seen, text, "import", file_id, content_version, parser_version, language,
                             line_start, line_end, ".".join(parents) or None, {"symbol_kind": kind})
            elif kind in _COMMENT_TYPES:
                self._append(blocks, seen, text, "comment", file_id, content_version, parser_version, language,
                             line_start, line_end, ".".join(parents) or None, {"symbol_kind": kind})
            elif kind in _IDENTIFIER_TYPES and text:
                self._append(blocks, seen, text, "identifier", file_id, content_version, parser_version, language,
                             line_start, line_end, ".".join(parents) or None, {"symbol_kind": kind})
            for child in node.children:
                visit(child, parents)

        visit(root, ())
        return blocks

    @staticmethod
    def _append(blocks, seen, text, content_type, file_id, content_version, parser_version, language, line_start, line_end, section, metadata):
        if not text:
            return
        key = (content_type, line_start, line_end)
        if key in seen:
            return
        seen.add(key)
        blocks.append(make_block(text, file_id=file_id, content_version=content_version,
                                 parser_name="tree-sitter", parser_version=parser_version,
                                 content_type=content_type, language=language, line_start=line_start,
                                 line_end=max(line_start, line_end), section_title=section, metadata=metadata))

    def _failed(self, file_id, content_version, error):
        return ExtractedDocument(file_id=file_id, content_version=content_version, status=ExtractionStatus.FAILED,
                                 blocks=[], parser_name=self.name, parser_version="unknown", error=error)
