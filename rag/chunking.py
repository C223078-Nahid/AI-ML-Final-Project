"""Text chunking with source-metadata preservation.

Chunks are sized in approximate tokens (about 4 characters per token for
English text).  Each chunk keeps the page number / section of the section it
was created from, so citations stay verifiable.  Consecutive chunks overlap
by roughly ``overlap_tokens`` to avoid cutting sentences apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.parsers import ParsedSection

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class TextChunk:
    text: str
    page_number: int | None
    section: str | None
    chunk_index: int
    token_estimate: int


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _tail_for_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return the trailing sentences that make up the overlap window."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        cost = estimate_tokens(sentence)
        if tail and total + cost > overlap_tokens:
            break
        tail.insert(0, sentence)
        total += cost
    return tail


def _hard_split(sentence: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """Split an oversized sentence on character boundaries with overlap."""
    max_chars = chunk_size_tokens * 4
    overlap_chars = max(0, overlap_tokens * 4)
    step = max(1, max_chars - overlap_chars)
    segments: list[str] = []
    start = 0
    while start < len(sentence):
        segments.append(sentence[start : start + max_chars])
        if start + max_chars >= len(sentence):
            break
        start += step
    return segments


def chunk_sections(
    sections: list[ParsedSection],
    chunk_size_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[TextChunk]:
    """Split parsed sections into overlapping chunks, preserving metadata.

    Content fidelity (follow-up task, section 3.2): a section that fits the
    chunk budget is kept **verbatim** - paragraph boundaries, line breaks,
    and key-value labels such as ``Name:`` survive exactly as uploaded.
    Only oversized sections fall back to sentence-split chunks (joined with
    newlines) with the usual overlap window.
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    raw: list[tuple[str, int | None, str | None]] = []

    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        if estimate_tokens(text) <= chunk_size_tokens:
            # Small section: one chunk, complete and verbatim.
            raw.append((text, section.page_number, section.section))
            continue

        sentences = _split_sentences(text)
        if not sentences:
            continue

        current: list[str] = []

        def flush() -> None:
            if current:
                # Newline join keeps a visual line structure even in the
                # oversized-section path (labels stay on their own lines).
                raw.append(("\n".join(current), section.page_number, section.section))
                current.clear()

        for sentence in sentences:
            cost = estimate_tokens(sentence)
            if cost > chunk_size_tokens:
                # Oversized sentence (e.g. minified text): hard-split it.
                flush()
                segments = _hard_split(sentence, chunk_size_tokens, overlap_tokens)
                for segment in segments[:-1]:
                    raw.append((segment, section.page_number, section.section))
                current.extend([segments[-1]] if segments else [])
                continue
            if current and estimate_tokens("\n".join(current)) + cost > chunk_size_tokens:
                overlap = _tail_for_overlap(current, overlap_tokens)
                flush()
                current.extend(overlap)
            current.append(sentence)
        flush()

    return [
        TextChunk(
            text=text,
            page_number=page_number,
            section=section,
            chunk_index=index,
            token_estimate=estimate_tokens(text),
        )
        for index, (text, page_number, section) in enumerate(raw)
    ]
