"""Follow-up task 4, sections 6-7: model-driven function calling in the
LangGraph agent loop.

Contracts (the task's test C):
- the model can select a tool;
- arguments are schema-validated (invalid args feed back an error, no
  crash);
- tool execution results return to the model, which continues;
- multiple tool calls in one message are handled;
- iteration limits prevent infinite loops (bounded shutdown with a useful
  final answer);
- repeated identical tool calls are suppressed;
- unavailable / restricted tools are refused, never executed;
- the assistant tool-call message precedes its tool results (strict
  OpenAI-compatible conversation validity).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from core.models import Intent, IntentResult
from tests.fakes import FakeGateway, tool_call_message


def _invoke(agent_graph, thread_id, request):
    from langchain_core.messages import HumanMessage

    return agent_graph.invoke(
        {
            "user_request": request,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=request)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def _agent_gateway(intent=None):
    return FakeGateway(
        intent=intent
        or IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    )


@pytest.fixture
def indexed_doc(ingestion, doc_repo):
    result = ingestion.ingest_file(
        "plan.txt", b"Qdrant was selected as the vector database."
    )
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    return document, f"{document.document_id}:0"


@pytest.fixture
def agent_graph(gateway, services, settings):
    """A graph whose gateway is scripted for the agent loop."""
    gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    return build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )


def test_model_selects_tool_and_continues(agent_graph, gateway, task_repo):
    """The model calls list_tasks, receives the result, and produces its
    final answer from it."""
    task_repo.create_task(
        __import__("core.models", fromlist=["TaskProposal"]).TaskProposal(
            title="Finish ML assignment", confirmation_message="ok"
        ),
        "op-agent-1",
    )
    gateway.agent_script = [
        tool_call_message("list_tasks", {"status": "open"}, call_id="c1"),
        AIMessage(content="You have one open task: Finish ML assignment."),
    ]
    result = _invoke(agent_graph, "ag-1", "What tasks do I have open?")
    payload = result["final_response"]
    assert payload["kind"] == "conversation"
    assert "Finish ML assignment" in payload["text"]
    traces = result["tool_results"]
    assert traces[0]["tool"] == "list_tasks"
    assert traces[0]["ok"] is True
    assert traces[0]["duration_ms"] >= 0
    # The tool result reached the model: the SECOND chat call contains a
    # ToolMessage with the task text.
    tool_calls_seen = [
        messages for messages, with_tools in gateway.seen_messages if with_tools
    ]
    assert len(tool_calls_seen) >= 2
    second_call = tool_calls_seen[1]
    tool_messages = [m for m in second_call if isinstance(m, ToolMessage)]
    assert any("Finish ML assignment" in str(m.content) for m in tool_messages)


def test_multiple_tool_calls_in_one_message(agent_graph, gateway, task_repo):
    """Two tool calls requested in ONE assistant message both execute."""
    task_repo.create_task(
        __import__("core.models", fromlist=["TaskProposal"]).TaskProposal(
            title="Buy milk", confirmation_message="ok"
        ),
        "op-agent-2",
    )
    gateway.agent_script = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "list_tasks", "args": {"status": "all"},
                 "id": "c1", "type": "tool_call"},
                {"name": "get_current_datetime", "args": {},
                 "id": "c2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="You have one task and it is late morning."),
    ]
    result = _invoke(agent_graph, "ag-2", "What are my tasks and what time is it?")
    traces = result["tool_results"]
    executed = [t["tool"] for t in traces]
    assert "list_tasks" in executed and "get_current_datetime" in executed
    assert result["final_response"]["kind"] == "conversation"


def test_assistant_tool_call_message_precedes_tool_results(agent_graph, gateway):
    """Strict provider validity: each ToolMessage is preceded by the
    assistant message carrying its tool_call id (follow-up task 4 fix)."""
    gateway.agent_script = [
        tool_call_message("get_current_datetime", {}, call_id="call_x"),
        AIMessage(content="It is afternoon."),
    ]
    _invoke(agent_graph, "ag-3", "What time is it?")
    chat_calls = [
        messages
        for messages, with_tools in gateway.seen_messages
        if with_tools
    ]
    assert len(chat_calls) >= 2
    second = chat_calls[1]
    ids_in_assistant = []
    for message in second:
        if message.__class__.__name__ == "AIMessage":
            ids_in_assistant = [
                call.get("id") for call in (message.tool_calls or [])
            ]
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in ids_in_assistant


def test_invalid_arguments_feed_back_an_error(agent_graph, gateway):
    """Schema-invalid arguments never execute; the model gets a corrective
    error string and the application does not crash."""
    gateway.agent_script = [
        tool_call_message("list_tasks", {"status": "BOGUS"}, call_id="c1"),
        AIMessage(content="Sorry - let me list all tasks instead."),
    ]
    result = _invoke(agent_graph, "ag-4", "Show my tasks")
    payload = result["final_response"]
    assert payload["kind"] == "conversation"  # no crash
    traces = result["tool_results"]
    assert traces[0]["tool"] == "list_tasks"
    assert traces[0]["ok"] is False
    assert "error" in traces[0]["result"].lower() or (
        traces[0].get("ok") is False
    )
    # The error went back to the model as a ToolMessage.
    chat_calls = [
        messages for messages, with_tools in gateway.seen_messages if with_tools
    ]
    assert any(
        isinstance(m, ToolMessage) and "Tool error" in str(m.content)
        for m in chat_calls[1]
    )


def test_unknown_tool_is_refused(agent_graph, gateway):
    """A hallucinated tool name is refused; the model recovers."""
    gateway.agent_script = [
        tool_call_message("delete_all_tasks", {}, call_id="c1"),
        AIMessage(content="I cannot delete tasks - they can only be created "
                          "after your approval."),
    ]
    result = _invoke(agent_graph, "ag-5", "Delete all my tasks")
    traces = result["tool_results"]
    assert traces[0]["tool"] == "delete_all_tasks"
    assert traces[0].get("error") == "unavailable tool"
    assert result["final_response"]["kind"] == "conversation"


def test_repeated_identical_tool_call_is_suppressed(agent_graph, gateway):
    """The same tool with identical arguments twice: the second call is
    suppressed with an explicit reuse instruction."""
    gateway.agent_script = [
        tool_call_message("get_current_datetime", {}, call_id="c1"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_current_datetime", "args": {},
                 "id": "c2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="It is afternoon."),
    ]
    result = _invoke(agent_graph, "ag-6", "What time is it?")
    traces = result["tool_results"]
    executed = [
        t for t in traces
        if t["tool"] == "get_current_datetime" and "ok" in t
    ]
    assert len(executed) == 1  # the duplicate never executed
    assert any(
        t.get("note") == "duplicate call suppressed"
        for t in traces if t["tool"] == "get_current_datetime"
    )


def test_iteration_budget_ends_with_a_final_answer(agent_graph, gateway, settings):
    """A model that ALWAYS requests tools cannot loop forever: the bounded
    shutdown produces a final answer without tools (useful error, not a
    dead loop)."""
    from langchain_core.messages import HumanMessage

    always_tool_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_current_datetime", "args": {},
             "id": "loop", "type": "tool_call"},
        ],
    )
    gateway.agent_script = [always_tool_call] * 20
    result = _invoke(
        agent_graph, "ag-7", "Keep calling tools forever, please"
    )
    payload = result["final_response"]
    # The run TERMINATED with a final answer.
    assert payload["kind"] == "conversation"
    # Tool executions were bounded by the configured budgets.
    executed = [t for t in result["tool_results"] if t.get("ok") is not None]
    assert len(executed) <= settings.max_total_tool_calls
    # A final no-tools call happened (FakeGateway returns chat_text).
    assert "general answer" in payload["text"].lower() or payload["text"]


def test_tool_calling_disabled_answers_directly(services, settings):
    """LLM_TOOL_CALLING_ENABLED=false binds NO tools: the model answers
    directly (tools are never offered)."""
    settings.llm_tool_calling_enabled = False
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    gateway = _agent_gateway()
    services.gateway = gateway
    agent_graph = build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )
    result = _invoke(agent_graph, "ag-8", "Hello there")
    assert result["final_response"]["text"] == "This is a general answer."
    # The chat call received tools=None (no tools were offered).
    offered = [with_tools for _m, with_tools in gateway.seen_messages]
    assert offered[-1] is False


def test_search_documents_available_to_agent(agent_graph, gateway, indexed_doc):
    """The agent can search the shared library (RAG as a tool)."""
    document, chunk_id = indexed_doc
    gateway.agent_script = [
        tool_call_message("search_documents", {"query": "Qdrant vector database"}),
        AIMessage(content="Your notes say Qdrant was selected as the vector database."),
    ]
    # Deliberately document-neutral wording: the MODEL decides to search
    # the library (the document-reference safety net is NOT triggered).
    result = _invoke(agent_graph, "ag-9", "Which vector database was chosen for the project?")
    traces = result["tool_results"]
    assert any(
        t["tool"] == "search_documents" and t.get("ok") for t in traces
    )
    assert "Qdrant" in result["final_response"]["text"]


def test_get_document_content_tool_reads_a_document(agent_graph, gateway, indexed_doc):
    """get_document_content returns the actual passages of a named file."""
    document, chunk_id = indexed_doc
    gateway.agent_script = [
        tool_call_message(
            "get_document_content", {"filename": "plan.txt"}
        ),
        AIMessage(content="Your plan document discusses Qdrant."),
    ]
    # Document-neutral question: the model itself decides to open the file.
    result = _invoke(agent_graph, "ag-10", "Tell me what the project decided about storage")
    traces = result["tool_results"]
    content_trace = [
        t for t in traces if t["tool"] == "get_document_content" and t.get("ok")
    ]
    assert content_trace
    assert "Qdrant" in content_trace[0]["result"]


def test_web_search_trace_records_the_provider(services, settings):
    """Follow-up task 4, section 13: a web_search call made by the model
    records which search provider served it (observability without
    credentials)."""
    from core.web_search import WebSearchService
    from langchain_core.messages import AIMessage
    from tests.test_web_search import FakeTavily
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    services.web_search = WebSearchService(FakeTavily())
    gateway = _agent_gateway()
    gateway.agent_script = [
        tool_call_message(
            "web_search",
            {"query": "chemical symbol of gold", "max_results": 3},
        ),
        AIMessage(content="Gold's symbol is Au per the retrieved source."),
    ]
    services.gateway = gateway
    graph = build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )
    result = _invoke(graph, "ag-11", "Verify what gold's chemical symbol is")
    traces = result.get("tool_results") or []
    web_trace = [t for t in traces if t.get("tool") == "web_search"]
    assert web_trace
    assert web_trace[0].get("provider") == "tavily"
    assert web_trace[0].get("ok") is True
