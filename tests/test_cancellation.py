"""Request-cancellation tests (follow-up task, section 7).

Covers the cooperative cancellation design end-to-end at the graph level
and the generation-job lifecycle:

A. Cancellation between nodes: every branch short-circuits to
   final_response, which emits an explicit stopped marker; the transcript
   stays coherent (user message + stopped marker, no dangling request, no
   duplicated messages).
B. Cancellation inside tool loops: no tool executes after Stop.
C. Cancellation before irreversible writes: task creation is prevented,
   the human-approval interrupt is never reached.
D. Checkpoint integrity: a cancelled run never corrupts state; the next
   message on the same thread works normally.
E. The generation job lifecycle: worker thread honours the cancellation
   event, reports real progress phases, and ends in a terminal state.
F. The pending-operation bookkeeping marks a cancelled approved run
   honestly (not "executed").
"""

from __future__ import annotations

import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from core import cancellation
from core.models import Intent, IntentResult, TaskProposal
from graph.nodes import STOPPED_TEXT
from tests.fakes import FakeGateway, tool_call_message

QUESTION = "What does my document say about Qdrant?"
TASK_REQUEST = "Remind me to review the architecture doc tomorrow"


def _ask(graph, thread_id, text):
    return graph.invoke(
        {
            "user_request": text,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=text)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def _messages(graph, thread_id):
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return list(state.values.get("messages") or [])


# ---------------------------------------------------------------------------
# A. Cancellation between nodes
# ---------------------------------------------------------------------------
def test_cancelled_run_emits_stopped_marker(services, settings):
    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    graph = settings and _build_graph(services, settings)
    thread_id = "cancel-marker"
    cancellation.begin_generation(threading.Event())

    # Cancel BEFORE the run starts: the very first nodes short-circuit.
    cancellation.cancellation_requested()  # sanity: context active
    event = threading.Event()
    event.set()
    cancellation.end_generation()
    cancellation.begin_generation(event)

    result = _ask(graph, thread_id, QUESTION)
    cancellation.end_generation()

    final = result["final_response"]
    assert final["kind"] == "stopped"
    assert final["stopped"] is True
    assert final["text"] == STOPPED_TEXT
    # Coherent transcript: user message + stopped marker, nothing else.
    messages = _messages(graph, thread_id)
    assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage"]
    assert messages[-1].additional_kwargs["pkta_payload"]["kind"] == "stopped"


def test_cancel_during_rag_generation_stops_before_answer(services, settings):
    """Stop pressed after routing but before retrieval: no retrieval
    happens, no model answer is generated, stopped marker is emitted."""
    from rag import diagnostics  # noqa: F401  (imports stay side-effect free)

    services.gateway.intent = IntentResult(
        intent=Intent.DOCUMENT_QUESTION,
        confidence=0.9,
        requires_retrieval=True,
    )
    graph = _build_graph(services, settings)
    thread_id = "cancel-rag"

    # Cancel while intent_router is executing (after input_guardrail).
    original = services.gateway.structured

    def _structured_then_cancel(schema, messages, run_name=None, metadata=None):
        result = original(schema, messages, run_name=run_name, metadata=metadata)
        # Simulate the user pressing Stop right after classification.
        event = threading.Event()
        event.set()
        cancellation.end_generation()
        cancellation.begin_generation(event)
        return result

    services.gateway.structured = _structured_then_cancel
    cancellation.begin_generation(threading.Event())
    try:
        result = _ask(graph, thread_id, QUESTION)
    finally:
        cancellation.end_generation()

    assert result["final_response"]["kind"] == "stopped"
    # No grounded answer was ever requested.
    assert "GroundedAnswer" not in services.gateway.structured_calls


def test_uncancelled_runs_unchanged(services, settings):
    """Sanity: without a cancellation context, every path behaves exactly
    as before (the thread-local flag defaults to no cancellation)."""
    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    graph = _build_graph(services, settings)
    result = _ask(graph, "normal-run", "Hello there")
    assert result["final_response"]["kind"] == "conversation"
    # The FakeGateway answers "DONE" when tools are bound (general_response
    # binds its read-only tools) - the point here is that the run completes
    # NORMALLY, without any stopped marker.
    assert not result["final_response"].get("stopped")


# ---------------------------------------------------------------------------
# B. Cancellation inside tool loops
# ---------------------------------------------------------------------------
def test_cancel_inside_general_response_tool_loop(services, settings):
    """The model requests tools; Stop is pressed after the FIRST tool call:
    no further tool executes and the run finishes as stopped."""
    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    # Script: first a tool call, then (would be) another tool call.
    services.gateway.agent_script = [
        tool_call_message(
            "search_documents", {"query": "qdrant"}, call_id="call_1"
        ),
        tool_call_message(
            "search_documents", {"query": "again"}, call_id="call_2"
        ),
    ]
    graph = _build_graph(services, settings)
    thread_id = "cancel-toolloop"

    stop_event = threading.Event()
    tool_executed = []

    # The agent loop builds its tools through the centralized registry
    # (follow-up task 4), whose factory imports the tool constructor from
    # tools.document_tools at call time - patch it THERE.
    import tools.document_tools as document_tools_module

    original_make = document_tools_module.make_search_documents_tool

    def _tracing_make(retriever, top_k, collector):
        tool = original_make(retriever, top_k, collector)

        class _Tracing:
            name = tool.name
            description = tool.description
            args_schema = getattr(tool, "args_schema", None)

            def invoke(self, args, config=None):
                tool_executed.append(args)
                if len(tool_executed) >= 1:
                    # Stop pressed right after the first tool executed.
                    stop_event.set()
                return tool.invoke(args, config)

        return _Tracing()

    document_tools_module.make_search_documents_tool = _tracing_make
    try:
        cancellation.begin_generation(stop_event)
        result = _ask(graph, thread_id, "Search my notes about Qdrant")
    finally:
        cancellation.end_generation()
        document_tools_module.make_search_documents_tool = original_make

    # Exactly ONE tool executed; the second pending call never ran.
    assert len(tool_executed) == 1
    assert result["final_response"]["kind"] == "stopped"


# ---------------------------------------------------------------------------
# C. Cancellation before irreversible writes
# ---------------------------------------------------------------------------
def test_cancelled_task_proposal_never_reaches_approval(services, settings):
    """Stop during task-proposal generation: no proposal, no interrupt, no
    pending operation - just the stopped marker."""
    services.gateway.intent = IntentResult(
        intent=Intent.CREATE_TASK, confidence=0.9, requires_tool=True
    )
    graph = _build_graph(services, settings)
    thread_id = "cancel-proposal"

    original = services.gateway.structured

    def _structured_then_cancel(schema, messages, run_name=None, metadata=None):
        result = original(schema, messages, run_name=run_name, metadata=metadata)
        event = threading.Event()
        event.set()
        cancellation.end_generation()
        cancellation.begin_generation(event)
        return result

    services.gateway.structured = _structured_then_cancel
    cancellation.begin_generation(threading.Event())
    try:
        result = _ask(graph, thread_id, TASK_REQUEST)
    finally:
        cancellation.end_generation()
        services.gateway.structured = original

    assert result["final_response"]["kind"] == "stopped"
    assert not result.get("task_proposal")
    assert not (result.get("__interrupt__") or ())


def test_cancelled_approved_run_does_not_create_task(
    services, settings, task_repo
):
    """The one irreversible write: a proposal is approved, but Stop is
    pressed before task_execution - the task is NOT created."""
    services.gateway.intent = IntentResult(
        intent=Intent.CREATE_TASK, confidence=0.9, requires_tool=True
    )
    services.gateway.proposal = TaskProposal(
        title="Review architecture doc",
        confirmation_message="Shall I create this task?",
    )
    graph = _build_graph(services, settings)
    thread_id = "cancel-before-write"

    cancellation.begin_generation(threading.Event())
    try:
        result = _ask(graph, thread_id, TASK_REQUEST)
    finally:
        cancellation.end_generation()

    interrupts = result.get("__interrupt__") or ()
    assert interrupts, "expected the approval interrupt for a healthy run"

    # The user approves... but then Stop is signalled before the write.
    event = threading.Event()
    event.set()
    cancellation.begin_generation(event)
    try:
        resumed = graph.invoke(
            Command(resume="approved"),
            {"configurable": {"thread_id": thread_id}},
        )
    finally:
        cancellation.end_generation()

    final = resumed["final_response"]
    assert final["kind"] == "stopped"
    # THE critical assertion: no task was written.
    assert task_repo.count_tasks() == 0


def test_uncancelled_approval_still_creates_task(services, settings, task_repo):
    """Sanity: without Stop, the approved proposal still writes the task
    (cancellation did not break the approval workflow)."""
    services.gateway.intent = IntentResult(
        intent=Intent.CREATE_TASK, confidence=0.9, requires_tool=True
    )
    services.gateway.proposal = TaskProposal(
        title="Review architecture doc",
        confirmation_message="Shall I create this task?",
    )
    graph = _build_graph(services, settings)
    thread_id = "approve-normal"
    result = _ask(graph, thread_id, TASK_REQUEST)
    assert (result.get("__interrupt__") or ()), "expected interrupt"
    resumed = graph.invoke(
        Command(resume="approved"), {"configurable": {"thread_id": thread_id}}
    )
    assert resumed["final_response"]["kind"] == "task_created"
    assert task_repo.count_tasks() == 1


# ---------------------------------------------------------------------------
# D. Checkpoint integrity after cancellation
# ---------------------------------------------------------------------------
def test_thread_is_reusable_after_cancelled_run(services, settings):
    """After a cancelled run, the next message on the same thread works
    normally and the transcript has no duplicated or dangling messages."""
    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    graph = _build_graph(services, settings)
    thread_id = "reuse-after-cancel"

    event = threading.Event()
    event.set()
    cancellation.begin_generation(event)
    try:
        _ask(graph, thread_id, "first question")
    finally:
        cancellation.end_generation()

    # A new message on the same thread - no cancellation this time.
    result = _ask(graph, thread_id, "second question")
    assert result["final_response"]["kind"] == "conversation"

    messages = _messages(graph, thread_id)
    kinds = [type(m).__name__ for m in messages]
    assert kinds == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    assert (
        messages[1].additional_kwargs["pkta_payload"]["kind"] == "stopped"
    )
    texts = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
    assert texts == ["first question", "second question"]


def test_cancel_state_is_reset_per_turn(services, settings):
    """input_guardrail resets ``cancelled``: a stale flag from a previous
    turn cannot cancel later turns."""
    from graph.routing import route_after_intent

    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    graph = _build_graph(services, settings)
    event = threading.Event()
    event.set()
    cancellation.begin_generation(event)
    _ask(graph, "stale-flag", "turn one")
    cancellation.end_generation()

    # No active cancellation context: the next turn must not see cancelled
    # state left over in the checkpoint.
    result = _ask(graph, "stale-flag", "turn two")
    assert result["final_response"]["kind"] == "conversation"


# ---------------------------------------------------------------------------
# E. Generation job lifecycle (the background worker)
# ---------------------------------------------------------------------------
class _SlowGateway(FakeGateway):
    """Gateway whose chat call blocks until released (simulates a model
    request in flight when Stop is pressed)."""

    def __init__(self):
        super().__init__()
        self.intent = IntentResult(
            intent=Intent.GENERAL_CONVERSATION, confidence=0.9
        )
        self.release = threading.Event()
        self.entered = threading.Event()

    def chat(self, messages, tools=None, run_name=None, metadata=None):
        self.entered.set()
        # Block like a real HTTP call the user wants to stop.
        self.release.wait(timeout=10)
        return super().chat(messages, tools, run_name, metadata)


def test_generation_job_lifecycle_cancel_during_model_call(
    services, settings
):
    """Stop pressed while the model call is in flight: the job survives the
    in-flight call, then winds down cleanly and reports CANCELLED."""
    slow = _SlowGateway()
    services.gateway = slow
    graph = _build_graph(services, settings)

    from ui import generation

    job_token = generation.start_generation(
        _FakeContainer(graph), "job-thread", "hello"
    )
    job = generation.get_job(job_token)
    assert job is not None and job.status == generation.RUNNING

    # Wait until the worker is inside the model call, then press Stop.
    assert slow.entered.wait(timeout=5)
    assert generation.request_stop(job_token)
    # The in-flight call is released (its result will be discarded).
    slow.release.set()

    deadline = time.time() + 10
    while job.status == generation.RUNNING and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == generation.CANCELLED
    assert job.result is not None
    final = job.result["final_response"]
    assert final["kind"] == "stopped"
    assert final["stopped"] is True
    # A terminal job can be disposed exactly once.
    generation.dispose_job(job_token)
    assert generation.get_job(job_token) is None


def test_generation_job_reports_real_progress_phases(services, settings):
    """The job's phase label tracks actual application events."""
    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    graph = _build_graph(services, settings)

    from ui import generation

    phases: list[str] = []
    token = generation.start_generation(
        _FakeContainer(graph), "phase-thread", "hello"
    )
    job = generation.get_job(token)
    # Hook the phase reporting to collect what the nodes actually report.
    original = job.report_phase

    def _collect(phase):
        phases.append(phase)
        original(phase)

    job.report_phase = _collect
    deadline = time.time() + 10
    while job.status == generation.RUNNING and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == generation.DONE
    assert job.result["final_response"]["kind"] == "conversation"
    # Real phases were reported (thinking at minimum).
    assert any("Thinking" in p for p in phases) or job.current_phase
    generation.dispose_job(token)


def test_request_stop_is_idempotent(services, settings):
    services.gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    graph = _build_graph(services, settings)
    from ui import generation

    token = generation.start_generation(
        _FakeContainer(graph), "idem-thread", "hi"
    )
    assert generation.request_stop(token) is True
    job = generation.get_job(token)
    deadline = time.time() + 10
    while job.status == generation.RUNNING and time.time() < deadline:
        time.sleep(0.05)
    # Terminal jobs cannot be stopped again.
    assert generation.request_stop(token) is False
    generation.dispose_job(token)


# ---------------------------------------------------------------------------
# F. Pending-operation bookkeeping for cancelled approved runs
# ---------------------------------------------------------------------------
def test_resume_operation_marks_cancelled_run_honestly(
    services, settings, conv_repo
):
    """resume_operation records a cancelled approved run as 'cancelled',
    not 'executed' (the task was never written)."""
    from ui.graph_service import resume_operation, send_message

    services.gateway.intent = IntentResult(
        intent=Intent.CREATE_TASK, confidence=0.9, requires_tool=True
    )
    services.gateway.proposal = TaskProposal(
        title="Review architecture doc",
        confirmation_message="Shall I create this task?",
    )
    graph = _build_graph(services, settings)
    container = _FakeContainer(graph, conv_repo=conv_repo)

    result = send_message(container, "bookkeeping-thread", TASK_REQUEST)
    payload = (result.get("__interrupt__") or [])[0].value
    operation_id = payload["operation_id"]

    # Stop is pressed before the approved write executes.
    event = threading.Event()
    event.set()
    cancellation.begin_generation(event)
    try:
        resumed = resume_operation(
            container, "bookkeeping-thread", operation_id, "approved"
        )
    finally:
        cancellation.end_generation()

    assert resumed["final_response"]["kind"] == "stopped"
    record = conv_repo.get_pending_operation(operation_id)
    assert record.status == "cancelled"


def test_resume_operation_unaffected_without_cancellation(
    services, settings, conv_repo, task_repo
):
    from ui.graph_service import resume_operation, send_message

    services.gateway.intent = IntentResult(
        intent=Intent.CREATE_TASK, confidence=0.9, requires_tool=True
    )
    services.gateway.proposal = TaskProposal(
        title="Buy oat milk",
        confirmation_message="Create this task?",
    )
    graph = _build_graph(services, settings)
    container = _FakeContainer(graph, conv_repo=conv_repo)

    result = send_message(container, "approve-thread", "Remind me to buy oat milk")
    payload = (result.get("__interrupt__") or [])[0].value
    operation_id = payload["operation_id"]
    resumed = resume_operation(
        container, "approve-thread", operation_id, "approved"
    )
    assert resumed["final_response"]["kind"] == "task_created"
    assert conv_repo.get_pending_operation(operation_id).status == "executed"
    assert task_repo.count_tasks() == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_graph(services, settings):
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    checkpointer = create_checkpointer(settings.checkpoint_db_path)
    return build_workflow(services, checkpointer)


class _FakeContainer:
    """The subset of the AppContainer the graph path needs."""

    def __init__(self, graph, conv_repo=None):
        self.graph = graph
        self.conversation_repo = conv_repo or _NullConvRepo()


class _NullConvRepo:
    def ensure_conversation(self, *a, **k):
        pass

    def touch(self, *a, **k):
        pass

    def record_pending_operation(self, *a, **k):
        pass

    def get_pending_operation(self, operation_id):
        return None

    def resolve_pending_operation(self, *a, **k):
        pass
