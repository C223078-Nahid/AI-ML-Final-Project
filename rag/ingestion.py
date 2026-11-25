"""Ingestion service: upload -> validate -> dedupe -> parse -> chunk -> embed -> store.

The service re-validates uploads (defense in depth) and records indexing
status and errors in the documents table so the Document Library can display
them.

Performance behaviour (fixes the reported slow-ingestion issue):
- **Stage timing** - every pipeline stage is measured with
  ``time.perf_counter`` and returned in ``IngestionResult.timings`` so the
  UI (and tests) can see exactly where time goes.
- **Content dedupe** - the SHA-256 of the file bytes is computed once per
  upload; if an indexed document with the same hash exists, ingestion
  returns ``status="duplicate"`` immediately without parsing, chunking, or
  embedding anything. This also makes Streamlit reruns (which re-trigger
  the uploader widget) effectively free.
- **Filename replacement** - uploading a *different* version under the same
  filename replaces the previous record and vectors instead of accumulating
  near-duplicate entries.
- **Batched embedding + batched Qdrant upserts** - see ``rag/vector_store``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable

from core import guardrails
from core.models import IngestionResult
from database.documents import DocumentRepository
from config.settings import Settings
from rag import diagnostics
from rag.chunking import chunk_sections
from rag.lexical_index import LexicalIndex
from rag.parsers import parse_document
from rag.vector_store import VectorStore

# Optional callback: progress(stage: str, detail: str) -> None.
# ``stage`` is one of: validating, deduplicating, parsing, chunking,
# embedding, indexing, done. The UI uses it to show real feedback.
ProgressFn = Callable[[str, str], None]


class _StageTimer:
    """Accumulates wall-clock seconds per named stage."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        elapsed = time.perf_counter() - self._starts.pop(stage, time.perf_counter())
        self.timings[stage] = self.timings.get(stage, 0.0) + elapsed
        return elapsed


def _noop_progress(stage: str, detail: str) -> None:  # noqa: ARG001
    return None


class IngestionService:
    def __init__(
        self,
        vector_store: VectorStore,
        document_repo: DocumentRepository,
        settings: Settings,
        lexical_index: LexicalIndex | None = None,
    ) -> None:
        self._store = vector_store
        self._documents = document_repo
        self._settings = settings
        self._lexical = lexical_index

    # ------------------------------------------------------------------
    def ingest_file(
        self,
        filename: str,
        data: bytes,
        progress: ProgressFn | None = None,
    ) -> IngestionResult:
        """Index one uploaded document. Never raises for user-level errors."""
        report = progress or _noop_progress
        timer = _StageTimer()
        started = time.perf_counter()

        def finish(**kwargs) -> IngestionResult:
            total = time.perf_counter() - started
            return IngestionResult(
                filename=filename,
                timings={**timer.timings, "total": total},
                duration_seconds=total,
                **kwargs,
            )

        # -- 1. file validation (extension + size) --------------------------
        report("validating", "Validating file...")
        timer.start("validate")
        extension = Path(filename).suffix.lower()
        ok, error = guardrails.validate_upload_file(
            filename, len(data), self._settings
        )
        timer.stop("validate")
        if not ok:
            return finish(document_id="", status="error", error_message=error)

        # -- 2. duplicate detection (cheap hash lookup) ---------------------
        report("deduplicating", "Checking for duplicates...")
        timer.start("dedupe")
        content_hash = hashlib.sha256(data).hexdigest()
        existing = self._documents.find_by_content_hash(content_hash)
        timer.stop("dedupe")
        if existing is not None:
            # Same bytes already indexed: skip everything. This is what makes
            # re-uploads and Streamlit reruns effectively free.
            report(
                "done",
                f"'{existing.filename}' is already indexed "
                f"({existing.chunk_count} chunks).",
            )
            return finish(
                document_id=existing.document_id,
                status="duplicate",
                chunk_count=existing.chunk_count,
                page_count=existing.page_count,
            )

        # -- 3. library capacity (only for genuinely new documents) ---------
        current_count = self._documents.count_documents(("indexed", "processing"))
        if current_count >= self._settings.max_documents:
            return finish(
                document_id="",
                status="error",
                error_message=(
                    f"The document library is full "
                    f"({self._settings.max_documents} documents). Remove a "
                    "document before uploading another one."
                ),
            )

        # -- 4. parse (before any destructive action: a corrupt upload must
        #    never remove an already-indexed version of the same filename) --
        report("parsing", "Extracting text...")
        timer.start("parse")
        try:
            parsed = parse_document(filename, data)
        except Exception as exc:  # noqa: BLE001 - recorded for the UI
            timer.stop("parse")
            report("error", f"Failed: {exc}")
            return finish(
                document_id="",
                status="error",
                error_message=str(exc),
            )
        timer.stop("parse")

        # -- 5. chunk ----------------------------------------------------
        report("chunking", "Generating chunks...")
        timer.start("chunk")
        chunks = chunk_sections(
            parsed.sections,
            chunk_size_tokens=self._settings.chunk_size_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )
        timer.stop("chunk")
        if not chunks:
            report("error", "Failed: no text chunks could be generated.")
            return finish(
                document_id="",
                status="error",
                error_message="No text chunks could be generated.",
            )

        # -- 6. register; a different file under the same filename replaces
        #    the old version instead of piling up near-duplicates ----------
        record = self._documents.register_document(filename, extension, content_hash)
        try:
            for older in self._documents.find_by_filename(filename):
                if older.document_id != record.document_id:
                    self._store.delete_document(older.document_id)
                    if self._lexical is not None:
                        self._lexical.delete_document(older.document_id)
                    self._documents.delete_document(older.document_id)

            # -- 7. embed -------------------------------------------------
            report("embedding", f"Generating embeddings for {len(chunks)} chunks...")
            timer.start("embed")
            vectors = self._store.embed_texts([chunk.text for chunk in chunks])
            timer.stop("embed")

            # -- 8. index ---------------------------------------------------
            report("indexing", "Indexing in Qdrant...")
            timer.start("index")

            def _qdrant_progress(_stage: str, done: int, total: int) -> None:
                report("indexing", f"Indexing in Qdrant... ({done}/{total} chunks)")

            stored = self._store.upsert_points(
                document_id=record.document_id,
                filename=filename,
                file_type=extension,
                uploaded_at=record.uploaded_at,
                chunks=chunks,
                vectors=vectors,
                progress=_qdrant_progress,
            )
            timer.stop("index")

            self._documents.mark_indexed(
                record.document_id, chunk_count=stored, page_count=parsed.page_count
            )
            # Keep the lexical (FTS5) half of hybrid retrieval in sync. The
            # Qdrant points and the FTS rows are written from the same chunk
            # list, so both stores describe exactly the same content.
            if self._lexical is not None:
                self._lexical.replace_document(record.document_id, filename, chunks)
            # Safe diagnostics: confirms the points actually landed in the
            # shared collection (task section 3.2.B). Never logs content.
            diagnostics.log_ingestion(
                self._store, filename, record.document_id, stored
            )
            pages = f", {parsed.page_count} pages" if parsed.page_count else ""
            report("done", f"Document ready ({stored} chunks{pages}).")
            return finish(
                document_id=record.document_id,
                status="indexed",
                chunk_count=stored,
                page_count=parsed.page_count,
            )
        except Exception as exc:  # noqa: BLE001 - recorded for the UI
            self._documents.mark_error(record.document_id, str(exc))
            report("error", f"Failed: {exc}")
            return finish(
                document_id=record.document_id,
                status="error",
                error_message=str(exc),
            )

    def delete_document(self, document_id: str) -> bool:
        """Remove a document from the index and the library table."""
        record = self._documents.delete_document(document_id)
        if record is None:
            return False
        if record.status == "indexed":
            self._store.delete_document(document_id)
            if self._lexical is not None:
                self._lexical.delete_document(document_id)
        return True
