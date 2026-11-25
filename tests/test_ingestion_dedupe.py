"""Ingestion performance and deduplication (follow-up task, section 2 & 11A).

Covers:
- duplicate uploads are skipped without re-parsing/re-embedding,
- per-stage timings are recorded and reported,
- a new version under the same filename replaces the old one,
- a corrupt upload never destroys an already-indexed version,
- re-uploading an indexed file still works when the library is full,
- databases created before the content_hash column are migrated in place.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag.ingestion import IngestionService


@pytest.fixture
def counting_embeddings(embeddings):
    """Fake embeddings that count how often batches are encoded."""
    calls = {"embed_documents": 0, "texts": 0}

    class _Counting(embeddings.__class__):
        def embed_documents(self, texts):
            calls["embed_documents"] += 1
            calls["texts"] += len(texts)
            return super().embed_documents(texts)

    counting = _Counting(dim=embeddings.dim)
    return counting, calls


@pytest.fixture
def shared_store(settings, counting_embeddings):
    """One vector store per test, wired to the counting embeddings.

    Both the ingestion service and the retriever must share the SAME
    QdrantClient instance: local-mode Qdrant refuses two clients on one
    storage folder.
    """
    from rag.vector_store import VectorStore

    counting, _ = counting_embeddings
    store = VectorStore(settings.qdrant_path, counting)
    yield store
    store.close()


@pytest.fixture
def shared_retriever(settings, shared_store):
    from rag.retrieval import Retriever

    return Retriever(shared_store, top_k=5, min_score=0.05)


@pytest.fixture
def perf_ingestion(settings, doc_repo, shared_store, counting_embeddings):
    counting, calls = counting_embeddings
    return IngestionService(shared_store, doc_repo, settings), calls


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------
def test_reupload_same_content_is_skipped(perf_ingestion, doc_repo):
    ingestion, counting = perf_ingestion
    data = b"The project uses PostgreSQL as its primary relational database."
    first = ingestion.ingest_file("notes.txt", data)
    assert first.status == "indexed"
    embeds_after_first = counting["embed_documents"]

    second = ingestion.ingest_file("notes.txt", data)
    assert second.status == "duplicate"
    assert second.document_id == first.document_id
    assert second.chunk_count == first.chunk_count
    # No re-embedding, no new record.
    assert counting["embed_documents"] == embeds_after_first
    assert doc_repo.count_documents(("indexed",)) == 1


def test_duplicate_upload_different_filename_same_content(perf_ingestion, doc_repo):
    ingestion, _ = perf_ingestion
    data = b"Identical bytes under another name."
    ingestion.ingest_file("a.txt", data)
    result = ingestion.ingest_file("copy.txt", data)
    assert result.status == "duplicate"
    assert doc_repo.count_documents(("indexed",)) == 1


def test_duplicate_upload_allowed_when_library_full(
    perf_ingestion, doc_repo, settings
):
    ingestion, _ = perf_ingestion
    for i in range(settings.max_documents):
        ingestion.ingest_file(f"doc{i}.txt", f"Content of document {i}.".encode())
    assert doc_repo.count_documents(("indexed",)) == settings.max_documents
    # Re-uploading an already indexed file adds nothing: must succeed.
    result = ingestion.ingest_file("doc0.txt", b"Content of document 0.")
    assert result.status == "duplicate"
    # A genuinely new file is still rejected.
    new = ingestion.ingest_file("extra.txt", b"One document too many.")
    assert new.status == "error"
    assert "full" in new.error_message


# ---------------------------------------------------------------------------
# Stage timings
# ---------------------------------------------------------------------------
def test_stage_timings_recorded(perf_ingestion):
    ingestion, _ = perf_ingestion
    result = ingestion.ingest_file(
        "timed.txt", b"A short document about timing."
    )
    assert result.status == "indexed"
    for stage in ("validate", "dedupe", "parse", "chunk", "embed", "index", "total"):
        assert stage in result.timings, f"missing timing for {stage}"
        assert result.timings[stage] >= 0.0
    assert result.duration_seconds == pytest.approx(
        result.timings["total"], abs=0.01
    )


def test_progress_callback_receives_all_stages(perf_ingestion):
    ingestion, _ = perf_ingestion
    stages: list[str] = []
    result = ingestion.ingest_file(
        "progress.txt",
        b"A document for progress reporting.",
        progress=lambda stage, detail: stages.append(stage),
    )
    assert result.status == "indexed"
    assert stages[0] == "validating"
    assert "deduplicating" in stages
    assert "parsing" in stages
    assert "chunking" in stages
    assert "embedding" in stages
    assert "indexing" in stages
    assert stages[-1] == "done"


# ---------------------------------------------------------------------------
# Filename replacement + safety
# ---------------------------------------------------------------------------
def test_new_version_same_filename_replaces_old(perf_ingestion, doc_repo, shared_retriever):
    retriever = shared_retriever
    ingestion, _ = perf_ingestion
    old = ingestion.ingest_file(
        "report.txt", b"Old report content about Alpha systems."
    )
    assert old.status == "indexed"
    assert retriever.search("Old report content about Alpha systems.")

    new = ingestion.ingest_file(
        "report.txt", b"New report content about Beta systems."
    )
    assert new.status == "indexed"
    # One record only, and it points to the new content.
    records = doc_repo.list_documents()
    assert len([r for r in records if r.filename == "report.txt"]) == 1
    assert retriever.search("New report content about Beta systems.")
    # The old content is gone from the index.
    assert retriever.search("Old report content about Alpha systems.") == []


def test_corrupt_upload_keeps_existing_version(perf_ingestion, doc_repo, shared_retriever):
    from tests.test_rag import make_pdf

    retriever = shared_retriever
    ingestion, _ = perf_ingestion
    good = ingestion.ingest_file(
        "spec.pdf", make_pdf(["A valid specification about database sharding."])
    )
    assert good.status == "indexed"

    # A text-less (image-only style) PDF under the same name must fail
    # WITHOUT removing the already indexed version: parsing happens before
    # any destructive step in the ingestion pipeline.
    empty_pdf = make_pdf([" "])  # no extractable text -> ParsingError
    failed = ingestion.ingest_file("spec.pdf", empty_pdf)
    assert failed.status == "error"
    # The previously indexed spec is untouched and still searchable.
    hits = retriever.search("A valid specification about database sharding.")
    assert hits and hits[0].filename == "spec.pdf"
    records = [r for r in doc_repo.list_documents() if r.status == "indexed"]
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Legacy database migration
# ---------------------------------------------------------------------------
def test_content_hash_migration_on_legacy_database(settings, tmp_path):
    """Databases created before content_hash existed must migrate in place."""
    from database.connection import Database
    from database.schema import init_schema

    legacy_path = str(tmp_path / "legacy.db")
    legacy = Database(legacy_path)
    with legacy.lock:
        legacy.connection.execute(
            """
            CREATE TABLE documents (
                document_id    TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                file_type      TEXT NOT NULL,
                status         TEXT NOT NULL,
                chunk_count    INTEGER NOT NULL DEFAULT 0,
                page_count     INTEGER,
                error_message  TEXT,
                uploaded_at    TEXT NOT NULL
            )
            """
        )
        legacy.connection.execute(
            "INSERT INTO documents VALUES ('d1','old.txt','.txt','indexed',"
            "3,NULL,NULL,'2026-01-01T00:00:00')"
        )
        legacy.connection.commit()
    legacy.close()

    # init_schema must add the column without touching existing rows.
    db = Database(legacy_path)
    init_schema(db)
    columns = {
        row[1] for row in db.connection.execute("PRAGMA table_info(documents)")
    }
    assert "content_hash" in columns
    row = db.connection.execute(
        "SELECT document_id, status FROM documents WHERE document_id='d1'"
    ).fetchone()
    assert row["status"] == "indexed"
    db.close()


def test_migration_is_idempotent(settings, db):
    """Running init_schema twice must not fail or duplicate columns."""
    from database.schema import init_schema

    init_schema(db)
    init_schema(db)
    columns = [
        row[1] for row in db.connection.execute("PRAGMA table_info(documents)")
    ]
    assert columns.count("content_hash") == 1


# ---------------------------------------------------------------------------
# Vector store batching
# ---------------------------------------------------------------------------
def test_upsert_points_batches_large_sets(settings, doc_repo, embeddings, monkeypatch):
    from rag.chunking import TextChunk
    from rag.vector_store import VectorStore

    store = VectorStore(settings.qdrant_path, embeddings)
    batches: list[int] = []
    original_upsert = store._client.upsert

    def spy_upsert(*args, **kwargs):
        batches.append(len(kwargs.get("points") or args[1]))
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store._client, "upsert", spy_upsert)

    chunks = [
        TextChunk(text=f"Chunk number {i} with some content.", page_number=None,
                  section=None, chunk_index=i, token_estimate=6)
        for i in range(150)
    ]
    vectors = store.embed_texts([c.text for c in chunks])
    stored = store.upsert_points(
        document_id="doc-batch", filename="big.txt", file_type=".txt",
        uploaded_at="now", chunks=chunks, vectors=vectors, batch_size=64,
    )
    assert stored == 150
    assert batches == [64, 64, 22]  # batched, not one giant request


def test_ensure_collection_checked_once(settings, embeddings, monkeypatch):
    from rag.vector_store import VectorStore

    store = VectorStore(settings.qdrant_path, embeddings)
    calls = {"collection_exists": 0}
    original = store._client.collection_exists

    def spy(name):
        calls["collection_exists"] += 1
        return original(name)

    monkeypatch.setattr(store._client, "collection_exists", spy)
    store.ensure_collection()
    store.ensure_collection()
    store.ensure_collection()
    assert calls["collection_exists"] == 1  # cached after the first check
