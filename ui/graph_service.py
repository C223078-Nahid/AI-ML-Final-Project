"""Interaction helpers between the Streamlit UI and the LangGraph workflow.

This module owns the operations that touch the graph from the UI:
- chat-session management (new chat, switching, deletion),
- sending a message (which may pause at human_approval),
- resuming a paused workflow with an explicit approve/reject decision,
- uploading documents into the shared library with rerun protection.

Conversation memory is isolated per thread (LangGraph checkpoints), while
the document library is global and shared across every thread by design.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def new_thread_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Thread state access
# ---------------------------------------------------------------------------
def get_thread_state(container, thread_id: str):
    """Return the checkpointed state snapshot for a thread (or None)."""
    try:
        return container.graph.get_state(thread_config(thread_id))
    except Exception:  # noqa: BLE001 - unknown thread id
        return None


def get_thread_messages(container, thread_id: str) -> list[Any]:
    """Return the persisted Human/AI messages of a conversation thread."""
    snapshot = get_thread_state(container, thread_id)
    if snapshot is None:
        return []
    return [
        m
        for m in (snapshot.values.get("messages") or [])
        if m.__class__.__name__ in ("HumanMessage", "AIMessage")
    ]


def pending_interrupt(container, thread_id: str) -> dict | None:
    """Return the pending task-approval interrupt payload, if any."""
    snapshot = get_thread_state(container, thread_id)
    if snapshot is None:
        return None
    if "human_approval" not in (snapshot.next or ()):
        return None
    proposal = snapshot.values.get("task_proposal") or {}
    operation_id = snapshot.values.get("operation_id") or ""
    if not proposal:
        return None
    return {
        "type": "task_approval",
        "operation_id": operation_id,
        "proposal": proposal,
    }


# ---------------------------------------------------------------------------
# Chat-session management
# ---------------------------------------------------------------------------
def send_message(container, thread_id: str, text: str) -> dict[str, Any]:
    """Run the workflow for a new user message.

    If the workflow pauses at human_approval, the pending proposal is
    recorded in the approval-center index and the interrupt payload is
    attached to the result.
    """
    from database.conversations import title_from_message

    container.conversation_repo.ensure_conversation(
        thread_id, title_from_message(text)
    )
    result = container.graph.invoke(
        {
            "user_request": text,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=text)],
        },
        thread_config(thread_id),
    )
    interrupts = result.get("__interrupt__") or ()
    if interrupts:
        payload = interrupts[0].value or {}
        if payload.get("type") == "task_approval":
            container.conversation_repo.record_pending_operation(
                payload["operation_id"], thread_id, payload["proposal"]
            )
        return result
    container.conversation_repo.touch(thread_id)
    return result


def resume_operation(
    container, thread_id: str, operation_id: str, decision: str
) -> dict[str, Any] | None:
    """Resume an interrupted workflow with an explicit approve/reject.

    Idempotency layers:
    1. The pending-operation index is resolved before the graph resumes, so
       a double click cannot process the same proposal twice.
    2. The graph only resumes if the checkpoint is still paused at
       human_approval.
    3. The tasks table enforces UNIQUE(operation_id).
    """
    if decision not in ("approved", "rejected"):
        decision = "rejected"
    pending = container.conversation_repo.get_pending_operation(operation_id)
    if pending is not None and pending.status != "pending":
        return None  # already resolved by an earlier click

    snapshot = get_thread_state(container, thread_id)
    if snapshot is None or "human_approval" not in (snapshot.next or ()):
        if pending is not None:
            container.conversation_repo.resolve_pending_operation(
                operation_id, "executed" if decision == "approved" else "rejected"
            )
        return None

    if pending is not None:
        container.conversation_repo.resolve_pending_operation(
            operation_id, decision
        )
    try:
        result = container.graph.invoke(
            Command(resume=decision), thread_config(thread_id)
        )
    except Exception:  # noqa: BLE001 - surface in the UI, keep index consistent
        if pending is not None:
            container.conversation_repo.resolve_pending_operation(
                operation_id, "failed"
            )
        raise
    final = (result or {}).get("final_response") or {}
    if final.get("kind") == "stopped":
        # The approved run was cancelled (Stop pressed) before the task
        # write: report it honestly instead of "executed".
        if pending is not None:
            container.conversation_repo.resolve_pending_operation(
                operation_id, "cancelled"
            )
        return result
    if decision == "approved" and pending is not None:
        container.conversation_repo.resolve_pending_operation(
            operation_id, "executed"
        )
    return result


# ---------------------------------------------------------------------------
# Shared document uploads (rerun-safe)
# ---------------------------------------------------------------------------
def upload_signature(name: str, size: int, data: bytes) -> str:
    """Stable signature of an uploaded file, used to skip reprocessing.

    Streamlit keeps the uploader widget's value across reruns; without this
    guard, every widget interaction would re-ingest the attached file. The
    signature combines filename, size, and a fast hash of the content.
    """
    import hashlib

    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{name}:{size}:{digest}"


def already_processed(session_state, signature: str) -> bool:
    processed = session_state.get("pkta_processed_uploads")
    if processed is None:
        processed = session_state["pkta_processed_uploads"] = set()
    return signature in processed


def mark_processed(session_state, signature: str) -> None:
    processed = session_state.get("pkta_processed_uploads")
    if processed is None:
        processed = session_state["pkta_processed_uploads"] = set()
    processed.add(signature)
    # Bound the set so a long session cannot grow it forever.
    if len(processed) > 200:
        session_state["pkta_processed_uploads"] = set(list(processed)[-100:])
