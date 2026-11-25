"""Qdrant vector store in local persistent mode.

No separately deployed Qdrant server is required: ``QdrantClient(path=...)``
embeds the storage engine and persists to disk.  Chunk payloads carry the
full citation metadata (document_id, filename, page, section) so citations
can always be traced back to real stored content.

Thread safety: qdrant-client's LOCAL mode has no internal locking, and
generation now runs in background worker threads (request cancellation,
follow-up task section 7) that share this store with the main Streamlit
thread.  Every client call is therefore serialized through a re-entrant
lock (``_LockedQdrant`` below) - correctness first; lock contention is
negligible for a single-user local application.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from core.models import RetrievedChunk
from rag.chunking import TextChunk

_COLLECTION = "documents"
# Deterministic point IDs make re-ingesting the same document idempotent.
_NAMESPACE = uuid.UUID("8b04a42b-9d1e-4f0f-9a44-6bb65746e6f8")


class _LockedQdrant:
    """Proxy that serializes every Qdrant client call behind one RLock.

    Local-mode qdrant-client is not thread-safe; the application shares one
    store between the Streamlit script thread (ingestion, stats) and
    background generation workers (retrieval).  The proxy keeps the change
    local to ``VectorStore`` - every existing caller keeps working
    unchanged.
    """

    def __init__(self, client: QdrantClient, lock: threading.RLock) -> None:
        self._client = client
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if callable(attribute):
            def _locked(*args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    return attribute(*args, **kwargs)

            return _locked
        return attribute


class VectorStore:
    def __init__(self, path: str, embeddings: Embeddings) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._client: Any = _LockedQdrant(QdrantClient(path=path), self._lock)
        self._embeddings = embeddings
        self._path = str(path)
        # ``ensure_collection`` is only allowed to touch Qdrant once per
        # process: the collection metadata cannot change while the app runs,
        # and re-checking it on every upload added avoidable latency.
        self._collection_ready = False
        self._dimension: int | None = None

    # -- collection management ------------------------------------------------
    @property
    def collection_name(self) -> str:
        return _COLLECTION

    @property
    def storage_path(self) -> str:
        """Local persistence directory of this store (for diagnostics)."""
        return self._path

    def stats(self) -> dict[str, Any]:
        """Non-sensitive snapshot of the shared collection (diagnostics).

        Used by the RAG diagnostics logs and the Settings page. Never
        includes document content or credentials.
        """
        info: dict[str, Any] = {
            "collection": _COLLECTION,
            "path": self._path,
            "exists": False,
            "points": 0,
            "dimension": self._dimension,
        }
        try:
            exists = self._client.collection_exists(_COLLECTION)
            info["exists"] = exists
            if exists:
                info["points"] = int(
                    self._client.count(_COLLECTION, exact=True).count
                )
        except Exception:  # noqa: BLE001 - diagnostics must not raise
            pass
        return info

    def ensure_collection(self) -> None:
        """Create the collection with the embedding dimension if missing.

        The result is cached for the lifetime of the store instance, so
        repeated uploads (and Streamlit reruns) do not re-issue collection
        metadata queries. Deleting the whole collection externally while the
        app runs is not a supported operation.
        """
        if self._collection_ready:
            return
        dimension = self._embedding_dimension()
        self._dimension = dimension
        if self._client.collection_exists(_COLLECTION):
            info = self._client.get_collection(_COLLECTION)
            existing = (
                info.config.params.vectors.size
                if info.config and info.config.params and info.config.params.vectors
                else None
            )
            if existing is not None and existing != dimension:
                raise RuntimeError(
                    "The existing vector index was built with a different "
                    f"embedding dimension ({existing} vs {dimension}). Remove "
                    f"the directory {self._client.path} or re-index."
                )
        else:
            self._client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
        self._collection_ready = True

    def _embedding_dimension(self) -> int:
        probe = self._embeddings.embed_query("dimension probe")
        return len(probe)

    # -- writes ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in one batched call (single model pass per batch)."""
        if not texts:
            return []
        return self._embeddings.embed_documents(texts)

    def upsert_points(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        uploaded_at: str,
        chunks: list[TextChunk],
        vectors: list[list[float]],
        *,
        batch_size: int = 64,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> int:
        """Store pre-computed chunk vectors, inserting in bounded batches.

        ``progress(stage, done, total)`` is called after every batch so the
        UI can show real ingestion feedback without fake percentages.
        Deterministic point IDs make re-ingesting the same document
        idempotent.
        """
        if not chunks:
            return 0
        self.ensure_collection()
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{document_id}:{chunk.chunk_index}")),
                vector=vector,
                payload={
                    "chunk_id": f"{document_id}:{chunk.chunk_index}",
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "text": chunk.text,
                    "uploaded_at": uploaded_at,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        total = len(points)
        for start in range(0, total, max(1, batch_size)):
            batch = points[start : start + max(1, batch_size)]
            self._client.upsert(collection_name=_COLLECTION, points=batch)
            if progress is not None:
                progress("indexing", min(start + len(batch), total), total)
        return total

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        uploaded_at: str,
        chunks: list[TextChunk],
    ) -> int:
        """Embed and store chunks in one call. Returns the number of points."""
        if not chunks:
            return 0
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return self.upsert_points(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            uploaded_at=uploaded_at,
            chunks=chunks,
            vectors=vectors,
        )

    def delete_document(self, document_id: str) -> None:
        """Remove every point belonging to a document."""
        self._client.delete(
            collection_name=_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def reconcile(self, valid_document_ids: set[str]) -> dict[str, int]:
        """Remove orphan points whose document no longer exists in the
        library database.

        A DB<->Qdrant desync (reported as "retrieval returned 2 chunks while
        the UI said 1 chunk") leaves points behind whose ``document_id`` has
        no indexed record. This scan-and-delete runs at application startup,
        which guarantees that Qdrant, the SQLite bookkeeping, and the UI
        chunk counts agree from then on.

        Returns ``{"scanned": n, "orphans_removed": m, "documents": d}``;
        never raises (best-effort cleanup, logged by the caller).
        """
        result = {"scanned": 0, "orphans_removed": 0, "documents": 0}
        try:
            if not self._client.collection_exists(_COLLECTION):
                return result
            orphans: set[str] = set()
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=_COLLECTION,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                )
                for point in points:
                    result["scanned"] += 1
                    payload: dict[str, Any] = point.payload or {}
                    doc_id = str(payload.get("document_id", ""))
                    if doc_id and doc_id not in valid_document_ids:
                        orphans.add(doc_id)
                if offset is None:
                    break
            for doc_id in orphans:
                self.delete_document(doc_id)
                result["orphans_removed"] += 1
            if result["scanned"]:
                result["points"] = int(
                    self._client.count(_COLLECTION, exact=True).count
                )
        except Exception:  # noqa: BLE001 - cleanup must not crash startup
            pass
        return result

    # -- reads ---------------------------------------------------------------------
    def document_chunks(
        self, document_id: str, limit: int = 8
    ) -> list[RetrievedChunk]:
        """Fetch a document's passages in chunk order (no similarity search).

        Used by the read-only get_document_content tool and the fact-check
        workflow: reading a specific document's content must not depend on
        embedding similarity to some query.  A missing collection means
        "nothing indexed" and returns an empty list.
        """
        if not self._client.collection_exists(_COLLECTION):
            return []
        points, _offset = self._client.scroll(
            collection_name=_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=max(1, limit),
            with_payload=True,
        )
        entries: list[tuple[int, RetrievedChunk]] = []
        for point in points:
            payload: dict[str, Any] = point.payload or {}
            text = payload.get("text") or ""
            if not text:
                continue
            chunk_id = str(payload.get("chunk_id", ""))
            try:
                order = int(chunk_id.rsplit(":", 1)[-1])
            except (ValueError, IndexError):
                order = 0
            entries.append(
                (
                    order,
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        document_id=str(payload.get("document_id", "")),
                        filename=str(payload.get("filename", "")),
                        page_number=payload.get("page_number"),
                        section=payload.get("section"),
                        text=text,
                        score=1.0,
                    ),
                )
            )
        entries.sort(key=lambda pair: pair[0])
        return [chunk for _order, chunk in entries]

    def search(
        self, query: str, top_k: int, min_score: float
    ) -> list[RetrievedChunk]:
        """Semantic search over the shared collection.

        Failure semantics (follow-up task, section 6):
        - A missing collection is NOT an error: it legitimately means
          "nothing indexed yet" and returns an empty result.
        - Any other failure (embedding model, storage) propagates so the
          workflow can report an infrastructure error instead of silently
          treating it as "no evidence found".
        """
        if not self._client.collection_exists(_COLLECTION):
            return []
        vector = self._embeddings.embed_query(query)
        response = self._client.query_points(
            collection_name=_COLLECTION,
            query=vector,
            limit=max(1, top_k),
            with_payload=True,
        )
        chunks: list[RetrievedChunk] = []
        for point in response.points:
            score = float(point.score) if point.score is not None else 0.0
            if score < min_score:
                continue
            payload: dict[str, Any] = point.payload or {}
            text = payload.get("text") or ""
            if not text:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", "")),
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    page_number=payload.get("page_number"),
                    section=payload.get("section"),
                    text=text,
                    score=score,
                )
            )
        return chunks

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - best effort
            pass
