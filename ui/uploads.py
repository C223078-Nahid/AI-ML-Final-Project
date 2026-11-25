"""Shared document-upload handling for the Chat and Documents pages.

One ingestion entry point for every upload surface (chat attachment button,
chat empty-state drop zone, Documents page uploader). Guarantees:

- staged, real progress feedback (no fake percentages),
- per-stage timing display (parse / chunk / embed / index),
- Streamlit-rerun safety (an upload is processed once, not on every rerun),
- duplicate uploads are detected and reported as already indexed.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from core.models import IngestionResult
from ui.graph_service import already_processed, mark_processed, upload_signature

STAGE_LABELS = {
    "validating": "Validating file",
    "deduplicating": "Checking for duplicates",
    "parsing": "Extracting text",
    "chunking": "Generating chunks",
    "embedding": "Generating embeddings",
    "indexing": "Indexing in Qdrant",
    "done": "Document ready",
    "error": "Failed",
}


def format_timings(result: IngestionResult) -> str:
    """Human-readable per-stage timing line for an ingestion result."""
    parts = []
    for stage in ("validate", "dedupe", "parse", "chunk", "embed", "index"):
        seconds = result.timings.get(stage)
        if seconds is not None and seconds >= 0.0005:
            parts.append(f"{stage} {seconds:.2f}s")
    total = result.timings.get("total", result.duration_seconds)
    parts.append(f"total {total:.2f}s")
    return " · ".join(parts)


def process_upload(container, filename: str, data: bytes) -> IngestionResult:
    """Ingest one file with live stage feedback. Idempotent across reruns."""
    signature = upload_signature(filename, len(data), data)
    if already_processed(st.session_state, signature):
        # Already handled in a previous run (Streamlit keeps the uploader
        # value across reruns): report the known outcome without redoing work.
        return IngestionResult(
            document_id="",
            filename=filename,
            status="duplicate",
        )
    mark_processed(st.session_state, signature)

    status = st.status(f"Uploading **{html.escape(filename)}**...", expanded=True)
    stages: dict[str, Any] = {}
    current_stage: list[str | None] = [None]

    def _mark_done(stage: str) -> None:
        placeholder = stages.get(stage)
        if placeholder is not None:
            placeholder.markdown(f"✓ {STAGE_LABELS.get(stage, stage)} — done")

    def progress(stage: str, detail: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        previous = current_stage[0]
        if previous is not None and previous != stage:
            _mark_done(previous)
        if stage in ("done", "error"):
            # The final status label communicates the outcome; no line needed.
            current_stage[0] = stage
            return
        if stage not in stages:
            stages[stage] = status.empty()
        stages[stage].markdown(f"**{label}** — {html.escape(detail)}")
        current_stage[0] = stage

    result = container.ingestion.ingest_file(filename, data, progress)

    previous = current_stage[0]
    if previous is not None and previous not in ("done", "error"):
        _mark_done(previous)

    if result.status == "indexed":
        pages = (
            f" · {result.page_count} pages" if result.page_count else ""
        )
        status.update(
            label=(
                f"✅ {result.filename} ready — {result.chunk_count} chunks"
                f"{pages} · {result.duration_seconds:.2f}s"
            ),
            state="complete",
            expanded=False,
        )
    elif result.status == "duplicate":
        status.update(
            label=f"ℹ️ {result.filename} is already indexed — skipped",
            state="complete",
            expanded=False,
        )
    else:
        status.update(
            label=f"❌ {result.filename} could not be indexed",
            state="error",
            expanded=True,
        )
        st.error(f"{result.filename}: {result.error_message}")
    st.caption(f"Stage timings — {format_timings(result)}")
    return result


def process_upload_list(container, files: list[Any]) -> list[IngestionResult]:
    """Process a list of Streamlit UploadedFile objects in order."""
    results = []
    for uploaded in files:
        results.append(
            process_upload(container, uploaded.name, uploaded.getvalue())
        )
    return results
