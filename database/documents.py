"""Document library repository: ingestion bookkeeping in SQLite.

The vector data itself lives in Qdrant; this table tracks filenames, file
types, indexing status, chunk counts, and ingestion errors so the Document
Library UI can render everything without scanning the vector store.
"""

from __future__ import annotations

from core.models import DocumentRecord, new_id, utc_now_iso
from database.connection import Database


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    @staticmethod
    def _row_to_record(row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            page_count=row["page_count"],
            error_message=row["error_message"],
            uploaded_at=row["uploaded_at"],
        )

    def register_document(
        self, filename: str, file_type: str, content_hash: str | None = None
    ) -> DocumentRecord:
        record = DocumentRecord(
            document_id=new_id(),
            filename=filename,
            file_type=file_type,
            status="processing",
            uploaded_at=utc_now_iso(),
        )
        self._db.execute_write(
            """
            INSERT INTO documents
                (document_id, filename, file_type, status, chunk_count,
                 page_count, error_message, uploaded_at, content_hash)
            VALUES (?, ?, ?, ?, 0, NULL, NULL, ?, ?)
            """,
            (
                record.document_id,
                record.filename,
                record.file_type,
                record.status,
                record.uploaded_at,
                content_hash,
            ),
        )
        return record

    def find_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        """Return the indexed document with exactly this content, if any.

        Used to make re-uploading the same file a cheap no-op instead of a
        full re-parse / re-embed / re-index cycle.
        """
        row = self._db.query_one(
            "SELECT * FROM documents WHERE content_hash = ? "
            "AND status = 'indexed' ORDER BY uploaded_at DESC",
            (content_hash,),
        )
        return self._row_to_record(row) if row else None

    def find_by_filename(self, filename: str) -> list[DocumentRecord]:
        """Return every non-error record stored under this filename."""
        rows = self._db.query_all(
            "SELECT * FROM documents WHERE filename = ? "
            "AND status != 'error' ORDER BY uploaded_at DESC",
            (filename,),
        )
        return [self._row_to_record(row) for row in rows]

    def mark_indexed(
        self, document_id: str, chunk_count: int, page_count: int | None
    ) -> None:
        self._db.execute_write(
            "UPDATE documents SET status = 'indexed', chunk_count = ?, "
            "page_count = ?, error_message = NULL WHERE document_id = ?",
            (chunk_count, page_count, document_id),
        )

    def mark_error(self, document_id: str, error_message: str) -> None:
        self._db.execute_write(
            "UPDATE documents SET status = 'error', error_message = ? "
            "WHERE document_id = ?",
            (error_message[:500], document_id),
        )

    def list_documents(self) -> list[DocumentRecord]:
        rows = self._db.query_all(
            "SELECT * FROM documents ORDER BY uploaded_at DESC, filename"
        )
        return [self._row_to_record(row) for row in rows]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        row = self._db.query_one(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        )
        return self._row_to_record(row) if row else None

    def count_documents(self, statuses: tuple[str, ...] = ("indexed",)) -> int:
        placeholders = ", ".join("?" for _ in statuses)
        row = self._db.query_one(
            f"SELECT COUNT(*) AS n FROM documents WHERE status IN ({placeholders})",
            tuple(statuses),
        )
        return int(row["n"]) if row else 0

    def delete_document(self, document_id: str) -> DocumentRecord | None:
        record = self.get_document(document_id)
        if record is None:
            return None
        self._db.execute_write(
            "DELETE FROM documents WHERE document_id = ?", (document_id,)
        )
        return record
