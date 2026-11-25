"""Document parsers (PDF / DOCX / Markdown / TXT).

Every parser returns :class:`ParseResult` with text sections plus the source
metadata (page number for PDFs, heading/section for DOCX and Markdown).
Parsers never invent metadata: formats without real page numbers return
``page_number=None`` and the citation layer reports the filename and section
instead.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path


class ParsingError(Exception):
    """Raised when a document cannot be parsed."""


@dataclass
class ParsedSection:
    text: str
    page_number: int | None = None
    section: str | None = None


@dataclass
class ParseResult:
    sections: list[ParsedSection] = field(default_factory=list)
    page_count: int | None = None


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CONTROL_CHARS.sub("", text).strip()


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def parse_document(filename: str, data: bytes) -> ParseResult:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return _parse_pdf(data)
    if extension == ".docx":
        return _parse_docx(data)
    if extension == ".md":
        result = _parse_markdown(_decode(data))
        return result
    if extension == ".txt":
        return _parse_txt(_decode(data))
    raise ParsingError(f"Unsupported file type: {extension or '(none)'}")


# ---------------------------------------------------------------------------
# PDF (page-accurate)
# ---------------------------------------------------------------------------
def _parse_pdf(data: bytes) -> ParseResult:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            raise ParsingError(
                "The PDF is password-protected and cannot be indexed."
            )
        sections: list[ParsedSection] = []
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - single bad page
                raise ParsingError(f"Failed to extract page {index + 1}: {exc}")
            text = _clean(text)
            if text:
                sections.append(ParsedSection(text=text, page_number=index + 1))
        if not sections:
            raise ParsingError(
                "No extractable text found. Scanned/image-only PDFs are not "
                "supported in this version."
            )
        return ParseResult(sections=sections, page_count=len(reader.pages))
    except ParsingError:
        raise
    except PdfReadError as exc:
        raise ParsingError(f"Invalid PDF file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ParsingError(f"Failed to parse PDF: {exc}") from exc


# ---------------------------------------------------------------------------
# DOCX (heading-aware, no page numbers)
# ---------------------------------------------------------------------------
def _parse_docx(data: bytes) -> ParseResult:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise ParsingError("python-docx is not installed") from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ParsingError(f"Invalid DOCX file: {exc}") from exc

    sections: list[ParsedSection] = []
    buffer: list[str] = []
    current_heading: str | None = None

    def flush() -> None:
        text = _clean("\n".join(part for part in buffer if part.strip()))
        if text:
            sections.append(
                ParsedSection(text=text, page_number=None, section=current_heading)
            )
        buffer.clear()

    for paragraph in document.paragraphs:
        style_name = (getattr(paragraph.style, "name", "") or "").lower()
        text = paragraph.text.strip()
        if style_name.startswith(("heading", "title")) and text:
            flush()
            current_heading = text
        elif text:
            buffer.append(text)
    flush()

    if not sections:
        raise ParsingError("No extractable text found in the DOCX file.")
    return ParseResult(sections=sections, page_count=None)


# ---------------------------------------------------------------------------
# Markdown (heading-aware, no page numbers)
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _parse_markdown(text: str) -> ParseResult:
    sections: list[ParsedSection] = []
    buffer: list[str] = []
    current_heading: str | None = None

    def flush() -> None:
        cleaned = _clean(
            _LINK_RE.sub(r"\1", _IMAGE_RE.sub(r"\1", "\n".join(buffer)))
        )
        if cleaned:
            sections.append(
                ParsedSection(text=cleaned, page_number=None, section=current_heading)
            )
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            flush()
            current_heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()

    if not sections:
        raise ParsingError("No extractable text found in the Markdown file.")
    return ParseResult(sections=sections, page_count=None)


# ---------------------------------------------------------------------------
# Plain text (single section, no page numbers, no headings)
# ---------------------------------------------------------------------------
def _parse_txt(text: str) -> ParseResult:
    cleaned = _clean(text)
    if not cleaned:
        raise ParsingError("The text file is empty.")
    return ParseResult(
        sections=[ParsedSection(text=cleaned, page_number=None, section=None)],
        page_count=None,
    )
