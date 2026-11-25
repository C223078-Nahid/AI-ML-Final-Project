"""Lexical (FTS5) half of hybrid document retrieval.

Why this exists (follow-up task, section 3.4): exact-fact questions such as
"What is my name?" or "What is my date of birth?" score only marginally with
pure vector similarity against a short personal-information chunk (measured
0.34-0.36 with all-MiniLM-L6-v2, barely above the 0.30 threshold). A lexical
keyword match on "name" / "date of birth" is deterministic and reliable for
exactly these cases.

Design constraints (from the task):
- Qdrant remains the semantic store; this index only SUPPLEMENTS it.
- No new retrieval framework: one SQLite FTS5 virtual table in the existing
  application database, written and pruned by the ingestion service.
- Results are deduplicated against the vector results by chunk_id in
  ``rag.retrieval``; identical text never reaches the model twice.

Safety: this module never logs chunk text. Queries are sanitized into
quoted FTS5 terms so user input can never inject FTS5 syntax.
"""

from __future__ import annotations

import re

from core.models import RetrievedChunk
from database.connection import Database
from rag.chunking import TextChunk

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# Terms that carry no signal for personal-fact retrieval.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "do", "does", "did",
    "what", "whats", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "my", "me", "i", "you", "your", "it", "its", "this", "that",
    "these", "those", "of", "in", "on", "at", "to", "for", "from", "with",
    "can", "could", "would", "should", "will", "tell", "about", "please",
    "hint", "be", "found",
}


def build_match_query(text: str, max_terms: int = 8) -> str:
    """Turn free-form user text into a safe quoted FTS5 OR-query.

    Each word becomes a quoted prefix term (``"name*"``), so "names" matches
    "name" and vice versa. FTS5 operators in the input can never execute:
    only ``[A-Za-z0-9_]+`` substrings are used.
    """
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    terms: list[str] = []
    for word in words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        if word not in terms:
            terms.append(word)
        if len(terms) >= max_terms:
            break
    if not terms:
        return ""
    return " OR ".join(f'"{term}"*' for term in terms)


class LexicalIndex:
    """SQLite FTS5 keyword index over the indexed chunks."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- writes -----------------------------------------------------------------
    def replace_document(
        self,
        document_id: str,
        filename: str,
        chunks: list[TextChunk],
    ) -> None:
        """(Re)index every chunk of one document (old rows removed first)."""
        with self._db.lock:
            self._db.connection.execute(
                "DELETE FROM chunks_fts WHERE document_id = ?", (document_id,)
            )
            self._db.connection.executemany(
                "INSERT INTO chunks_fts (text, chunk_id, document_id, filename,"
                " page_number, section) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.text,
                        f"{document_id}:{chunk.chunk_index}",
                        document_id,
                        filename,
                        chunk.page_number,
                        chunk.section,
                    )
                    for chunk in chunks
                ],
            )
            self._db.connection.commit()

    def delete_document(self, document_id: str) -> None:
        with self._db.lock:
            self._db.connection.execute(
                "DELETE FROM chunks_fts WHERE document_id = ?", (document_id,)
            )
            self._db.connection.commit()

    def reconcile(self) -> int:
        """Remove rows whose document is no longer indexed.

        The FTS table lives in the same database as the ``documents``
        bookkeeping table, so the validity check is a single subquery and
        cannot desync. Returns the number of rows removed; never raises.
        """
        try:
            with self._db.lock:
                cursor = self._db.connection.execute(
                    "DELETE FROM chunks_fts WHERE document_id NOT IN"
                    " (SELECT document_id FROM documents WHERE status = 'indexed')"
                )
                self._db.connection.commit()
            return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        except Exception:  # noqa: BLE001 - cleanup must not crash startup
            return 0

    def count_rows(self, document_id: str | None = None) -> int:
        with self._db.lock:
            if document_id is None:
                row = self._db.connection.execute(
                    "SELECT COUNT(*) FROM chunks_fts"
                ).fetchone()
            else:
                row = self._db.connection.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE document_id = ?",
                    (document_id,),
                ).fetchone()
        return int(row[0]) if row else 0

    # -- reads --------------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Keyword search ranked by bm25. Returns chunks WITHOUT a score
        that is comparable to cosine similarity; the retriever merges them
        below the vector results, so lexical hits bypass ``min_score`` by
        design (an exact keyword match is relevant regardless of the
        embedding distance)."""
        match = build_match_query(query)
        if not match:
            return []
        sql = (
            "SELECT chunk_id, document_id, filename, page_number, section, text"
            " FROM chunks_fts WHERE chunks_fts MATCH ?"
            " ORDER BY bm25(chunks_fts) LIMIT ?"
        )
        with self._db.lock:
            try:
                rows = self._db.connection.execute(
                    sql, (match, max(1, top_k))
                ).fetchall()
            except Exception as exc:  # noqa: BLE001 - FTS must never break retrieval
                # Logged (never raised): a silent [] here would masquerade as
                # "no evidence found" - the exact failure mode the follow-up
                # task forbids.
                import logging

                logging.getLogger("pkta.rag").error(
                    "[rag-diag] lexical index query failed: %s: %s",
                    type(exc).__name__, str(exc)[:200],
                )
                return []
        chunks: list[RetrievedChunk] = []
        for row in rows:
            text = row[5] or ""
            if not text:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(row[0] or ""),
                    document_id=str(row[1] or ""),
                    filename=str(row[2] or ""),
                    page_number=row[3],
                    section=row[4],
                    text=text,
                    score=None,
                )
            )
        return chunks
