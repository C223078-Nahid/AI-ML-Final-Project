"""Shared document library across chat sessions (follow-up task, sections 4 & 11B).

The document index is ONE global Qdrant collection: uploading a document in
one conversation must make it retrievable in every other conversation, while
conversation histories stay strictly isolated per thread.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from core.models import (
    Citation,
    GroundedAnswer,
    Intent,
    IntentResult,
)
from tests.fakes import FakeGateway, tool_call_message

DISTINCTIVE = "The project uses PostgreSQL as its primary relational database."


@pytest.fixture
def shared_library(ingestion, doc_repo):
    """One document indexed once - the 'uploaded in Chat 1' stand-in."""
    result = ingestion.ingest_file("ProjectNotes.txt", DISTINCTIVE.encode())
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    return document, f"{document.document_id}:0"


@pytest.fixture
def doc_gateway(gateway, shared_library):
    """Script the shared fake gateway for grounded document answers."""
    document, chunk_id = shared_library
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95)
    gateway.grounded = GroundedAnswer(
        answer=f"According to your document: {DISTINCTIVE}",
        citations=[
            Citation(
                document_id=document.document_id,
                filename="ProjectNotes.txt",
                chunk_id=chunk_id,
                supporting_passage=DISTINCTIVE,
            )
        ],
        confidence=0.9,
    )
    return gateway


def _rescript_planner(gateway):
    """Re-arm the planner's tool call (the script pops after each use).

    The fake embeddings only match near-identical text, so the retrieval
    planner is scripted to query with the exact chunk text - standing in
    for a real model formulating a good keyword query.
    """
    gateway.agent_script = [
        tool_call_message("search_documents", {"query": DISTINCTIVE})
    ]


def _ask(graph, thread_id: str, question: str):
    return graph.invoke(
        {
            "user_request": question,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=question)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def test_document_uploaded_in_chat1_retrievable_in_chat2_and_chat3(
    graph_factory, doc_gateway, shared_library
):
    """The exact scenario from the requirements.

    Chat 1 uploads; Chats 2 and 3 (different threads, empty histories) both
    retrieve the same content and produce grounded, cited answers.
    """
    document, chunk_id = shared_library

    # Thread A ("Chat 2"): fresh thread asking the question.
    _rescript_planner(doc_gateway)
    result_a = _ask(graph_factory(), "chat-2", "What database does the project use?")
    payload_a = result_a["final_response"]

    # Thread B ("Chat 3"): another fresh thread asking for a summary.
    _rescript_planner(doc_gateway)
    result_b = _ask(
        graph_factory(), "chat-3", "Summarize what my uploaded document says."
    )
    payload_b = result_b["final_response"]

    for payload in (payload_a, payload_b):
        assert payload["kind"] == "answer"
        assert payload["citations"], "no citations - content did not reach the model"
        assert payload["citations"][0]["chunk_id"] == chunk_id
        assert payload["citations"][0]["filename"] == "ProjectNotes.txt"


def test_retrieved_content_reaches_model_in_every_thread(
    graph_factory, doc_gateway
):
    """Stronger connectivity check: the chunk TEXT is inside the grounded
    answer prompt for every thread (the model actually sees the content)."""
    seen_by_thread: dict[str, bool] = {}
    for thread_id in ("t-x", "t-y", "t-z"):
        _rescript_planner(doc_gateway)
        before = len(doc_gateway.seen_messages)
        _ask(graph_factory(), thread_id, "What database does the project use?")
        found = any(
            DISTINCTIVE in str(getattr(m, "content", ""))
            for messages, _with_tools in doc_gateway.seen_messages[before:]
            for m in messages
        )
        seen_by_thread[thread_id] = found
    assert seen_by_thread == {"t-x": True, "t-y": True, "t-z": True}


def test_deleting_document_removes_it_for_every_chat(
    graph_factory, doc_gateway, shared_library, ingestion
):
    """Removal from the shared library must affect every thread."""
    document, _chunk_id = shared_library
    graph = graph_factory()

    # Available in a fresh thread before deletion.
    _rescript_planner(doc_gateway)
    result = _ask(graph, "before-delete", "What database does the project use?")
    assert result["final_response"]["citations"]

    ingestion.delete_document(document.document_id)

    # Now no thread can retrieve it; the assistant abstains.
    result2 = _ask(graph, "after-delete", "What database does the project use?")
    payload = result2["final_response"]
    assert payload["insufficient_evidence"] is True
    assert payload["citations"] == []


def test_conversation_histories_isolated_but_documents_shared(
    graph_factory, doc_gateway, shared_library
):
    """Messages must not leak between threads; document access must."""
    graph = graph_factory()
    _ask(graph, "iso-1", "Remember the codename Banana Republic.")
    state_1 = graph.get_state({"configurable": {"thread_id": "iso-1"}})
    _rescript_planner(doc_gateway)
    _ask(graph, "iso-2", "What database does the project use?")
    state_2 = graph.get_state({"configurable": {"thread_id": "iso-2"}})

    texts_1 = [str(m.content) for m in state_1.values["messages"]]
    texts_2 = [str(m.content) for m in state_2.values["messages"]]
    assert any("Banana" in t for t in texts_1)
    assert not any("Banana" in t for t in texts_2), "history leaked across threads"
    # But thread 2 still got its grounded answer with a citation.
    assert state_2.values["final_response"]["citations"]


def test_qdrant_collection_is_global_not_per_thread(shared_library):
    """The vector store exposes exactly one shared collection for all chats."""
    from rag.vector_store import _COLLECTION

    assert _COLLECTION == "documents"
