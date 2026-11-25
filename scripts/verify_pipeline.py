"""End-to-end pipeline verification with REAL local embeddings.

Runs the actual ingestion pipeline (real sentence-transformers model, real
Qdrant local storage, real LangGraph workflow) with a scripted gateway
standing in for the language model, so the whole RAG path can be verified
offline:

  1. Ingests small TXT / PDF / DOCX / Markdown documents and one multi-page
     document, measuring every stage (validate, dedupe, parse, chunk, embed,
     index) and printing a timing table.
  2. Verifies the embedding model is loaded exactly once across uploads.
  3. Verifies duplicate re-upload is skipped without re-embedding.
  4. Verifies the distinctive-statement scenario: upload in Chat 1, ask in
     Chats 2 and 3 - retrieved content must reach the answer prompt in every
     thread (checked by capturing what the model receives).
  5. Verifies abstention for a question the documents cannot answer.

Usage:
    python scripts/verify_pipeline.py [--keep] [--data-dir PATH]

Writes to a temporary data directory unless --data-dir is given; nothing in
your real ./data directory is touched. Requires the pinned dependencies
(including torch + sentence-transformers) and downloads the embedding model
from Hugging Face on first run (~90 MB).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402

from config.settings import Settings  # noqa: E402
from core.models import (  # noqa: E402
    Citation,
    GroundedAnswer,
    Intent,
    IntentResult,
)
from database.connection import Database  # noqa: E402
from database.conversations import ConversationRepository  # noqa: E402
from database.documents import DocumentRepository  # noqa: E402
from database.memory import MemoryRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from database.tasks import TaskRepository  # noqa: E402
from graph.nodes import GraphServices  # noqa: E402
from graph.persistence import create_checkpointer  # noqa: E402
from graph.workflow import build_workflow  # noqa: E402
from rag.embeddings import LocalEmbeddings  # noqa: E402
from rag.ingestion import IngestionService  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402
from tests.fakes import FakeGateway  # noqa: E402

DISTINCTIVE = "The project uses PostgreSQL as its primary relational database."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_pdf(pages: list[str]) -> bytes:
    """Minimal valid PDF with one text line per page (same as tests)."""
    from tests.test_rag import make_pdf as _make_pdf

    return _make_pdf(pages)


def make_docx(paragraphs: list[tuple[str, str]]) -> bytes:
    """Build a small DOCX from (style, text) pairs."""
    import io

    import docx

    document = docx.Document()
    for style, text in paragraphs:
        if style == "h1":
            document.add_heading(text, level=1)
        else:
            document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Reuse/inspect this data directory instead of a fresh temp one.",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the temporary data directory."
    )
    args = parser.parse_args()

    temp_dir = None
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        temp_dir = tempfile.mkdtemp(prefix="pkta-verify-")
        data_dir = Path(temp_dir)

    print("=" * 72)
    print("Pipeline verification (real embeddings, real Qdrant, real graph)")
    print(f"data directory: {data_dir}")
    print("=" * 72)

    settings = Settings(
        _env_file=str(PROJECT_ROOT / ".env") if (PROJECT_ROOT / ".env").exists()
        else None,
        data_dir=str(data_dir),
        qdrant_path=str(data_dir / "qdrant"),
        sqlite_path=str(data_dir / "app.db"),
        checkpoint_db_path=str(data_dir / "checkpoints.db"),
    )
    settings.ensure_directories()

    db = Database(settings.sqlite_path)
    init_schema(db)
    doc_repo = DocumentRepository(db)
    task_repo = TaskRepository(db)
    memory_repo = MemoryRepository(db)
    conv_repo = ConversationRepository(db)

    print("\n[1] Loading the embedding model once (warmup)...")
    t0 = time.perf_counter()
    embeddings = LocalEmbeddings(settings.embedding_model)
    embeddings.warmup()
    print(
        f"    model {settings.embedding_model} ready in "
        f"{time.perf_counter() - t0:.2f}s (dimension {embeddings.dimension})"
    )

    store = VectorStore(settings.qdrant_path, embeddings)
    retriever = Retriever(
        store, top_k=settings.retrieval_top_k, min_score=settings.retrieval_min_score
    )
    ingestion = IngestionService(store, doc_repo, settings)

    # ------------------------------------------------------------------ 2
    print("\n[2] Ingesting documents (stage timings in seconds)...")
    multi_page_pdf = make_pdf(
        [f"Page {i}: filler content about topic {i}." for i in range(1, 13)]
        + [f"Important page 13: {DISTINCTIVE}"]
        + [f"Page {i}: more filler about subject {i}." for i in range(14, 26)]
    )
    documents: list[tuple[str, bytes]] = [
        ("Small.txt", f"Overview. {DISTINCTIVE}".encode()),
        ("Small.pdf", make_pdf([f"Notes. {DISTINCTIVE}"])),
        ("Small.docx", make_docx([("h1", "Project"), ("p", DISTINCTIVE)])),
        ("Small.md", f"# Notes\n\n{DISTINCTIVE}\n".encode()),
        ("BigDocument.pdf", multi_page_pdf),
    ]
    print(f"    {'file':<18}{'status':<10}{'chunks':>6}{'pages':>7}"
          f"{'parse':>7}{'chunk':>7}{'embed':>7}{'index':>7}{'total':>8}")
    results = {}
    for name, data in documents:
        result = ingestion.ingest_file(name, data)
        results[name] = result
        t = result.timings
        print(
            f"    {name:<18}{result.status:<10}{result.chunk_count:>6}"
            f"{result.page_count or 0:>7}"
            f"{t.get('parse', 0):>7.3f}{t.get('chunk', 0):>7.3f}"
            f"{t.get('embed', 0):>7.3f}{t.get('index', 0):>7.3f}"
            f"{t.get('total', 0):>8.3f}"
        )
    small_total = results["Small.txt"].timings.get("total", 0)
    check(
        "small documents indexed quickly",
        small_total < 5.0,
        f"TXT total {small_total:.2f}s (first upload includes model warm-up)",
    )
    check(
        "embedding model loaded exactly once",
        embeddings.load_count == 1,
        f"load_count={embeddings.load_count}",
    )

    # ------------------------------------------------------------------ 3
    print("\n[3] Duplicate re-upload...")
    dup = ingestion.ingest_file("Small.txt", documents[0][1])
    check(
        "duplicate skipped without re-embedding",
        dup.status == "duplicate" and embeddings.load_count == 1,
        f"status={dup.status}, total={dup.timings.get('total', 0):.3f}s",
    )
    check(
        "no duplicate records",
        doc_repo.count_documents(("indexed",)) == len(documents),
        f"{doc_repo.count_documents(('indexed',))} records",
    )

    # ------------------------------------------------------------------ 4
    print("\n[4] Cross-chat RAG (upload in Chat 1, ask in Chats 2 & 3)...")
    gateway = FakeGateway()
    document = next(
        r for r in doc_repo.list_documents() if r.filename == "Small.txt"
    )
    chunk_id = f"{document.document_id}:0"
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95)
    gateway.grounded = GroundedAnswer(
        answer=f"According to your document: {DISTINCTIVE}",
        citations=[
            Citation(
                document_id=document.document_id,
                filename="Small.txt",
                chunk_id=chunk_id,
                supporting_passage=DISTINCTIVE,
            )
        ],
        confidence=0.9,
    )
    services = GraphServices(
        gateway=gateway,
        retriever=retriever,
        task_repo=task_repo,
        memory_repo=memory_repo,
        conversation_repo=conv_repo,
        settings=settings,
        doc_repo=doc_repo,
    )
    checkpointer = create_checkpointer(settings.checkpoint_db_path)
    graph = build_workflow(services, checkpointer)

    questions = {
        "chat-2": "What database does the project use according to my documents?",
        "chat-3": "Can you summarize what my uploaded document says about the database?",
    }
    for thread_id, question in questions.items():
        before = len(gateway.seen_messages)
        graph.invoke(
            {
                "user_request": question,
                "thread_id": thread_id,
                "messages": [HumanMessage(content=question)],
            },
            {"configurable": {"thread_id": thread_id}},
        )
        reached = any(
            DISTINCTIVE in str(getattr(m, "content", ""))
            for messages, _ in gateway.seen_messages[before:]
            for m in messages
        )
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        payload = state.values.get("final_response") or {}
        cited = bool(payload.get("citations"))
        check(
            f"{thread_id}: retrieved content reached the model",
            reached,
            "distinctive text present in the answer prompt",
        )
        check(
            f"{thread_id}: grounded answer with valid citation",
            cited and payload.get("citations", [{}])[0].get("chunk_id") == chunk_id,
            f"kind={payload.get('kind')}",
        )

    # ------------------------------------------------------------------ 5
    print("\n[5] Abstention on unsupported questions...")
    gateway.grounded = GroundedAnswer(
        answer="not found", citations=[], confidence=0.4, insufficient_evidence=True
    )
    state = graph.invoke(
        {
            "user_request": "What is the airspeed velocity of an unladen swallow?",
            "thread_id": "chat-4",
            "messages": [HumanMessage(
                content="What is the airspeed velocity of an unladen swallow?"
            )],
        },
        {"configurable": {"thread_id": "chat-4"}},
    )
    payload = state["final_response"]
    check(
        "assistant abstains without fabricating",
        payload.get("insufficient_evidence") is True and not payload.get("citations"),
        f"kind={payload.get('kind')}",
    )

    # ------------------------------------------------------------------ 6
    print("\n[6] Conversation isolation with shared documents...")
    t1 = graph.get_state({"configurable": {"thread_id": "chat-2"}})
    t2 = graph.get_state({"configurable": {"thread_id": "chat-3"}})
    t1_texts = [str(m.content) for m in t1.values["messages"]]
    t2_texts = [str(m.content) for m in t2.values["messages"]]
    check(
        "histories isolated across threads",
        not any("summarize what my uploaded document" in t.lower() for t in t1_texts)
        and not any("according to my documents" in t.lower() for t in t2_texts),
    )
    check(
        "both threads answered from the shared library",
        t1.values["final_response"]["citations"]
        and t2.values["final_response"]["citations"],
    )

    # ------------------------------------------------------------------ done
    print("\n" + "=" * 72)
    failures = [r for r in _results if r[0] == FAIL]
    for status, name, detail in _results:
        print(f"{status}  {name}" + (f" ({detail})" if detail else ""))
    print("=" * 72)
    print(f"{len(_results) - len(failures)}/{len(_results)} checks passed")

    store.close()
    db.close()
    if temp_dir and not args.keep:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("(temporary data directory removed; use --keep to inspect it)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
