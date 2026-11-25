"""E2 + C5: durable pause/resume, human approval, and idempotency."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.models import Intent, IntentResult, TaskPriority, TaskProposal
from ui.graph_service import (
    get_thread_state,
    pending_interrupt,
    resume_operation,
    send_message,
)

PROPOSAL = TaskProposal(
    title="Review lecture notes",
    description="Chapter 4 on neural networks",
    due_date=None,
    priority=TaskPriority.HIGH,
    confirmation_message="I will create the task 'Review lecture notes'.",
)


@pytest.fixture
def create_gateway(gateway):
    gateway.intent = IntentResult(intent=Intent.CREATE_TASK, confidence=0.9)
    gateway.proposal = PROPOSAL
    return gateway


@pytest.fixture
def container(graph, conv_repo, task_repo):
    """The object shape used by ui.graph_service helpers."""
    return SimpleNamespace(
        graph=graph, conversation_repo=conv_repo, task_repo=task_repo
    )


def _request(graph, thread_id: str, text: str = "Add a task to review lecture notes"):
    return graph.invoke(
        {
            "user_request": text,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=text)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Pause behavior
# ---------------------------------------------------------------------------
def test_pending_proposal_creates_no_task(graph, create_gateway, task_repo):
    result = _request(graph, "t-pending")
    interrupts = result.get("__interrupt__") or ()
    assert interrupts, "workflow must pause at human_approval"
    payload = interrupts[0].value
    assert payload["proposal"]["title"] == "Review lecture notes"
    assert payload["operation_id"]
    assert task_repo.count_tasks() == 0


def test_interrupt_payload_is_json_serializable(graph, create_gateway):
    import json

    result = _request(graph, "t-json")
    payload = result["__interrupt__"][0].value
    assert json.loads(json.dumps(payload))["proposal"]["title"]


def test_pending_operation_recorded_for_approval_center(
    graph, create_gateway, container, conv_repo
):
    send_message(container, "t-record", "Add a task to review lecture notes")
    pending = conv_repo.list_pending_operations("pending")
    assert len(pending) == 1
    assert pending[0].proposal["title"] == "Review lecture notes"
    assert pending[0].thread_id == "t-record"


def test_missing_approval_never_executes_task_tool(graph, create_gateway, task_repo):
    """Silence (no resume) must not count as approval."""
    _request(graph, "t-silent")
    snapshot = graph.get_state(_config("t-silent"))
    assert "human_approval" in snapshot.next
    assert task_repo.count_tasks() == 0
    assert task_repo.get_by_operation_id(
        (snapshot.values.get("operation_id") or "")
    ) is None


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------
def test_approval_creates_exactly_one_task(graph, create_gateway, task_repo, conv_repo):
    result = _request(graph, "t-approve")
    operation_id = result["__interrupt__"][0].value["operation_id"]
    conv_repo.record_pending_operation(operation_id, "t-approve", result["__interrupt__"][0].value["proposal"])

    resumed = graph.invoke(Command(resume="approved"), _config("t-approve"))
    payload = resumed["final_response"]
    assert payload["kind"] == "task_created"
    assert task_repo.count_tasks() == 1
    stored = task_repo.list_tasks()[0]
    assert stored.title == "Review lecture notes"
    assert stored.priority.value == "high"
    assert stored.operation_id == operation_id


def test_rejection_creates_no_task(graph, create_gateway, task_repo):
    _request(graph, "t-reject")
    resumed = graph.invoke(Command(resume="rejected"), _config("t-reject"))
    payload = resumed["final_response"]
    assert payload["kind"] == "task_rejected"
    assert "No task has been created" in payload["text"]
    assert task_repo.count_tasks() == 0


def test_ambiguous_resume_value_is_rejected(graph, create_gateway, task_repo):
    _request(graph, "t-ambig")
    resumed = graph.invoke(Command(resume="maybe, I guess"), _config("t-ambig"))
    assert resumed["final_response"]["kind"] == "task_rejected"
    assert task_repo.count_tasks() == 0


# ---------------------------------------------------------------------------
# Idempotency and durability
# ---------------------------------------------------------------------------
def test_double_approve_creates_no_duplicate(
    graph, create_gateway, task_repo, conv_repo, container
):
    send_message(container, "t-double", "Add a task to review lecture notes")
    pending = conv_repo.list_pending_operations("pending")[0]

    resume_operation(container, "t-double", pending.operation_id, "approved")
    assert task_repo.count_tasks() == 1
    # Second click on the same proposal is ignored (already resolved).
    resume_operation(container, "t-double", pending.operation_id, "approved")
    assert task_repo.count_tasks() == 1


def test_resume_after_graph_restart_keeps_single_task(
    graph_factory, create_gateway, task_repo, conv_repo
):
    """Leave a proposal pending, 'restart' the app, then resume it."""
    graph = graph_factory()
    result = _request(graph, "t-restart")
    operation_id = result["__interrupt__"][0].value["operation_id"]
    conv_repo.record_pending_operation(operation_id, "t-restart", result["__interrupt__"][0].value["proposal"])

    # Fresh graph instance on the same checkpoint database (app restart).
    restarted = graph_factory()
    snapshot = restarted.get_state(_config("t-restart"))
    assert "human_approval" in snapshot.next
    assert snapshot.values["task_proposal"]["title"] == "Review lecture notes"

    resumed = restarted.invoke(Command(resume="approved"), _config("t-restart"))
    assert resumed["final_response"]["kind"] == "task_created"
    assert task_repo.count_tasks() == 1

    # Resuming the completed thread again must not duplicate the task.
    try:
        restarted.invoke(Command(resume="approved"), _config("t-restart"))
    except Exception:
        pass  # some LangGraph versions reject resumes of finished threads
    assert task_repo.count_tasks() == 1


def test_repository_idempotency_on_operation_id(task_repo):
    first = task_repo.create_task(PROPOSAL, "op-dedupe-1")
    second = task_repo.create_task(PROPOSAL, "op-dedupe-1")
    assert first.created is True
    assert second.created is False
    assert first.task.id == second.task.id
    assert task_repo.count_tasks() == 1


def test_pending_interrupt_detection_helper(container, create_gateway, conv_repo):
    send_message(container, "t-helper", "Add a task to review lecture notes")
    pending = pending_interrupt(container, "t-helper")
    assert pending is not None
    assert pending["proposal"]["title"] == "Review lecture notes"
    # After rejection there is no pending interrupt anymore.
    resume_operation(container, "t-helper", pending["operation_id"], "rejected")
    assert pending_interrupt(container, "t-helper") is None
