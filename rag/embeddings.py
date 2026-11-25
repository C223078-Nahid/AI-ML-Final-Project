"""Embedding providers.

``LocalEmbeddings`` runs sentence-transformers locally (no embedding API
charges).  ``DeterministicFakeEmbeddings`` is a dependency-free fake used by
the automated tests so the test suite never downloads models or calls paid
APIs.

Performance notes (verified against the reported slow-ingestion issue):
- The SentenceTransformer model is loaded lazily ONCE per instance and then
  reused for every upload and query; ``warmup()`` lets the application
  pre-load it in the background at startup so the first upload is not paying
  the model-load cost.
- ``embed_documents`` encodes the whole batch in a single call with a bounded
  internal batch size instead of one network/GPU pass per text.
"""

from __future__ import annotations

import hashlib
import threading

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Sentence-Transformers embeddings running fully locally."""

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._model = None
        self._dim: int | None = None
        # Guards lazy model loading: the app warms the model up in a
        # background thread while uploads may arrive concurrently.
        self._lock = threading.Lock()
        self._load_count = 0

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._load_count += 1
            return self._model

    def warmup(self) -> int:
        """Load the model now (idempotent). Returns the embedding dimension.

        Callers can poll ``is_warm`` to show initialization progress in the
        UI while this runs in a background thread.
        """
        model = self._ensure_model()
        if self._dim is None:
            # sentence-transformers >= 5 renamed the accessor; support both.
            getter = getattr(
                model, "get_embedding_dimension", None
            ) or model.get_sentence_embedding_dimension
            self._dim = int(getter())
        return self._dim

    @property
    def is_warm(self) -> bool:
        return self._model is not None and self._dim is not None

    @property
    def load_count(self) -> int:
        """How many times the model was actually loaded (for diagnostics)."""
        return self._load_count

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class DeterministicFakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for tests.

    Identical texts produce identical vectors (cosine similarity 1.0), which
    lets the tests verify vector insertion, retrieval, and metadata without
    any model download.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            values.extend(byte / 255.0 for byte in digest)
            counter += 1
        vector = [value - 0.5 for value in values[: self.dim]]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
