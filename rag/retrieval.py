"""Hybrid retrieval façade used by the search_documents tool and the RAG node.

Hybrid design (follow-up task, section 3.4):

1. **Semantic** retrieval through Qdrant (unchanged, the primary path).
2. **Lexical** retrieval through the SQLite FTS5 index - a lightweight,
   deterministic keyword fallback that reliably finds exact facts
   ("Name:", "Date of Birth:") whose embedding similarity is marginal
   (measured 0.34-0.36 with all-MiniLM-L6-v2, close to the 0.30 threshold).
3. **Deduplication**: results are merged by chunk_id and by normalized text,
   so the same passage is never passed to the model twice and chunk counts
   stay consistent with the library.

The shared document library remains global: no thread or conversation
filter is ever applied here.
"""

from __future__ import annotations

from core.models import RetrievedChunk
from rag.lexical_index import LexicalIndex
from rag.vector_store import VectorStore


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int,
        min_score: float,
        lexical_index: LexicalIndex | None = None,
    ) -> None:
        self._store = vector_store
        self._lexical = lexical_index
        self.top_k = top_k
        self.min_score = min_score

    @property
    def store(self) -> VectorStore:
        """The shared vector store this retriever queries (diagnostics)."""
        return self._store

    @property
    def lexical_index(self) -> LexicalIndex | None:
        return self._lexical

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Hybrid search: Qdrant semantics first, FTS5 keywords as backup.

        Ordering of the merged result: vector hits (by cosine score,
        threshold-gated) come first, then lexical hits the vector search did
        not already return (keyword matches are relevant by construction and
        deliberately bypass ``min_score``).
        """
        limit = top_k if top_k is not None and top_k > 0 else self.top_k
        query_clean = " ".join(query.split())
        if not query_clean:
            return []

        vector_hits = self._store.search(query_clean, limit, self.min_score)

        if self._lexical is None:
            merged = list(vector_hits)
        else:
            lexical_hits = self._lexical.search(query_clean, limit)
            seen_ids = {chunk.chunk_id for chunk in vector_hits}
            seen_text = {_norm(chunk.text) for chunk in vector_hits}
            extras = [
                chunk
                for chunk in lexical_hits
                if chunk.chunk_id not in seen_ids
                and _norm(chunk.text) not in seen_text
            ]
            merged = vector_hits + extras

        # Text-level dedup: a Qdrant/SQLite desync can momentarily surface
        # the same passage under two ids; the model must never see it twice.
        deduped: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in merged:
            key = _norm(chunk.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped[: max(1, limit)]

    def document_chunks(self, document_id: str, limit: int = 8) -> list[RetrievedChunk]:
        """Read a document's passages directly (no similarity search).

        Read-only accessor for the get_document_content tool and the
        fact-check workflow: reading a specific document must not depend on
        a query.  Delegates to the vector store's chunk-order scroll.
        """
        return self._store.document_chunks(document_id, limit=limit)
