"""Chat-session management (follow-up task, sections 6 & 11C).

Covers the conversation registry: title generation from the first message,
title persistence, deletion semantics (documents untouched), and the
interaction between chat deletion and pending approvals.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from database.conversations import title_from_message
from tests.fakes import FakeGateway
from core.models import Intent, IntentResult


# ---------------------------------------------------------------------------
# Title generation (LLM-free by design)
# ---------------------------------------------------------------------------
def test_title_from_short_message():
    assert title_from_message("Hello there") == "Hello there"


def test_title_truncates_on_word_boundary():
    title = title_from_message(
        "What are the main topics covered in the architecture specification?"
    )
    assert len(title) <= 49  # 48 chars + ellipsis
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")  # cut at a word boundary


def test_title_short_messages_kept_verbatim():
    assert (
        title_from_message("What are the main topics in the PDF?")
        == "What are the main topics in the PDF?"
    )


def test_title_from_blank_message_is_none():
    assert title_from_message("") is None
    assert title_from_message("   \n\t ") is None


def test_title_collapses_whitespace():
    assert title_from_message("  Summarize   my\n\nnotes  ") == "Summarize my notes"


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------
def test_title_set_once_from_first_message(conv_repo):
    conv_repo.ensure_conversation("t1", title_from_message("First question here"))
    conv_repo.ensure_conversation("t1", title_from_message("Second message"))
    conv_repo.touch("t1")
    record = conv_repo.get_conversation("t1")
    assert record.title == "First question here"


def test_set_title_explicit_rename(conv_repo):
    conv_repo.ensure_conversation("t1", "old title")
    conv_repo.set_title("t1", "renamed chat")
    assert conv_repo.get_conversation("t1").title == "renamed chat"


def test_list_conversations_orders_by_recent_activity(conv_repo):
    conv_repo.ensure_conversation("a", "Chat A")
    conv_repo.ensure_conversation("b", "Chat B")
    conv_repo.touch("a")  # a becomes most recent
    assert [c.thread_id for c in conv_repo.list_conversations()] == ["a", "b"]


def test_delete_conversation_keeps_documents_and_other_chats(
    conv_repo, doc_repo, ingestion
):
    result = ingestion.ingest_file("keep.txt", b"Shared library content.")
    assert result.status == "indexed"
    conv_repo.ensure_conversation("keep-me", "Kept chat")
    conv_repo.ensure_conversation("delete-me", "Doomed chat")

    deleted = conv_repo.delete_conversation("delete-me")

    assert deleted is not None and deleted.thread_id == "delete-me"
    assert conv_repo.get_conversation("delete-me") is None
    assert conv_repo.get_conversation("keep-me") is not None
    # The shared document library is untouched by chat deletion.
    assert doc_repo.count_documents(("indexed",)) == 1
    assert deleted.title == "Doomed chat"


def test_delete_missing_conversation_returns_none(conv_repo):
    assert conv_repo.delete_conversation("ghost") is None


# ---------------------------------------------------------------------------
# Thread lifecycle through the graph
# ---------------------------------------------------------------------------
def _send(graph, thread_id: str, text: str):
    return graph.invoke(
        {
            "user_request": text,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=text)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def test_new_chat_gets_clean_context_and_old_chats_keep_theirs(
    graph_factory, gateway
):
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    graph = graph_factory()
    _send(graph, "old", "My favorite course is databases.")
    state_old = graph.get_state({"configurable": {"thread_id": "old"}})

    # A brand-new thread (new chat button) starts empty.
    state_new = graph.get_state({"configurable": {"thread_id": "brand-new"}})
    assert state_new.values == {} or not state_new.values.get("messages")
    assert len(state_old.values["messages"]) >= 2  # original exchange kept


def test_new_chat_does_not_clear_checkpoints_or_preferences(
    graph_factory, gateway, memory_repo
):
    memory_repo.save_preference("preferred_response_style", "concise")
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    graph = graph_factory()
    _send(graph, "t-a", "Hello from thread A")

    # Simulate pressing New Chat: a different thread id with no effect on
    # existing state.
    _send(graph_factory(), "t-b", "Hello from thread B")

    state_a = graph.get_state({"configurable": {"thread_id": "t-a"}})
    assert state_a.values["messages"], "checkpoints were cleared"
    prefs = memory_repo.list_preferences()
    assert any(p.key == "preferred_response_style" for p in prefs)


def test_pending_approval_survives_conversation_deletion(
    graph, gateway, conv_repo
):
    """A deleted conversation's pending approval stays resumable (checkpoints
    are intentionally preserved; deleting a chat must not delete documents)."""
    from types import SimpleNamespace
    from datetime import date, timedelta

    from core.models import TaskPriority, TaskProposal
    from ui.graph_service import send_message

    proposal = TaskProposal(
        title="Review deleted-chat task",
        description="",
        due_date=date.today() + timedelta(days=3),
        priority=TaskPriority.MEDIUM,
        confirmation_message="I will create the task 'Review deleted-chat task'.",
    )
    gateway.intent = IntentResult(intent=Intent.CREATE_TASK, confidence=0.9)
    gateway.proposal = proposal

    container = SimpleNamespace(
        graph=graph, conversation_repo=conv_repo
    )
    result = send_message(container, "doomed", "Add a task to review the notes")
    interrupts = result.get("__interrupt__") or ()
    assert interrupts, "expected the workflow to pause for approval"

    conv_repo.delete_conversation("doomed")

    # The approval index and the checkpoint both survive deletion.
    pending = conv_repo.list_pending_operations("pending")
    assert any(op.thread_id == "doomed" for op in pending)
    snapshot = graph.get_state({"configurable": {"thread_id": "doomed"}})
    assert "human_approval" in (snapshot.next or ())
