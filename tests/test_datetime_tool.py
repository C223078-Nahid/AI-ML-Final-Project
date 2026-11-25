"""Follow-up task 4, section 7: current date/time through MODEL-DRIVEN
function calling.

Contracts (the task's test B):
- "What day is today?" -> the MODEL decides to call get_current_datetime;
- the tool returns the actual configured-timezone clock (never training
  data, never documents);
- "What time is it in my timezone?" -> the same tool;
- fabricated model timestamps are never accepted (bounded corrective retry,
  then a deterministic application-clock footer);
- the old deterministic datetime graph node is gone.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from core.models import Intent, IntentResult
from tests.fakes import FakeGateway, tool_call_message

_FROZEN_NOW = datetime(2026, 11, 25, 15, 7, 0, tzinfo=ZoneInfo("Asia/Dhaka"))


def _build(services, settings):
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    return build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )


def _ask(graph, thread_id, question):
    from langchain_core.messages import HumanMessage

    return graph.invoke(
        {
            "user_request": question,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=question)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch):
    import core.datetime_util as dt_util

    monkeypatch.setattr(dt_util, "now", lambda tz=None: _FROZEN_NOW)


def test_tool_returns_the_application_clock():
    """The tool exposes date, time, weekday, timezone and ISO 8601 stamp."""
    import json

    from tools.datetime_tools import make_datetime_tool

    output = make_datetime_tool().invoke({})
    payload = json.loads(output)
    assert payload["date"] == "2026-11-25"
    assert payload["time"] == "15:07:00"
    assert payload["weekday"] == "Wednesday"
    assert payload["timezone"] == "Asia/Dhaka"
    assert payload["iso_8601"].startswith("2026-11-25T15:07:00")
    assert payload["source"] == "application clock"
    # Human-readable rendering included for natural answers.
    assert "25 November 2026" in payload["date_text"]


def test_model_decides_to_call_datetime_tool(services, settings):
    """"What day is today?" - the model invokes the tool and answers from
    its result (the exact expected interaction from the task)."""
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.CURRENT_DATETIME, confidence=0.9),
        agent_script=[
            tool_call_message("get_current_datetime", {}),
            AIMessage(content=(
                "Today is Wednesday, 25 November 2026. The current time is "
                "15:07 (BST, Asia/Dhaka)."
            )),
        ],
    )
    graph = _build(services, settings)
    result = _ask(graph, "dt-1", "What day is today?")
    payload = result["final_response"]

    assert payload["kind"] == "conversation"
    traces = result.get("tool_results") or []
    datetime_trace = [t for t in traces if t.get("tool") == "get_current_datetime"]
    assert len(datetime_trace) == 1
    # The tool ran the REAL utility: the frozen application clock.
    assert datetime_trace[0]["result"].__contains__("2026-11-25")
    assert "Wednesday" in datetime_trace[0]["result"]
    # The final answer is grounded in the tool output.
    assert "25 November 2026" in payload["text"]


def test_timezone_time_question_uses_the_same_tool(services, settings):
    """"What time is it in my timezone?" - same tool, real value."""
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9),
        agent_script=[
            tool_call_message("get_current_datetime", {}),
            AIMessage(content="It is currently 15:07 (Asia/Dhaka)."),
        ],
    )
    graph = _build(services, settings)
    result = _ask(graph, "dt-2", "What time is it in my timezone?")
    traces = result.get("tool_results") or []
    assert any(
        t.get("tool") == "get_current_datetime"
        and "15:07:00" in t.get("result", "")
        for t in traces
    )
    assert "15:07" in result["final_response"]["text"]


def test_fabricated_timestamp_is_not_accepted(services, settings):
    """The model answers a date question WITHOUT calling the tool: one
    corrective retry happens, and the deterministic application-clock value
    is appended so no fabricated date survives (section 7: never accept
    fabricated model timestamps)."""
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.CURRENT_DATETIME, confidence=0.9),
        chat_text="Today is Monday, 3 March 1997, 04:00 UTC.",  # fabricated
    )
    graph = _build(services, settings)
    result = _ask(graph, "dt-3", "What day is today?")
    payload = result["final_response"]

    # The corrective footer supplies the REAL clock value.
    assert "25 November 2026" in payload["text"]
    assert "application clock" in payload["text"]
    # The retry message explicitly instructed the model to call the tool.
    retry_calls = [
        messages
        for messages, with_tools in services.gateway.seen_messages
        if with_tools
    ]
    assert any(
        "get_current_datetime" in str([m.content for m in messages][-1])
        for messages in retry_calls
    )


def test_no_deterministic_datetime_node_in_the_graph(services, settings):
    """The hardcoded datetime routing is removed (adapted per section 7):
    current_datetime routes to the tool-calling agent instead."""
    graph = _build(services, settings)
    assert "datetime_response" not in getattr(graph, "nodes", {})
    from graph.routing import INTENT_ROUTES

    assert INTENT_ROUTES["current_datetime"] == "general_response"
    assert "fact_check" in INTENT_ROUTES


def test_datetime_never_answered_from_documents(services, settings, ingestion,
                                                 doc_repo):
    """A current-date question inside a document thread still never touches
    document retrieval (the original bug stays fixed)."""
    result = ingestion.ingest_file(
        "Testing.txt",
        b"Personal Informations:\n\nName: Alex Example\n\n"
        b"Date of Birth: 12-03-1995\n",
    )
    assert result.status == "indexed"
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9),
        agent_script=[
            tool_call_message("get_current_datetime", {}),
            AIMessage(content="Today is Wednesday, 25 November 2026."),
        ],
    )
    graph = _build(services, settings)
    result = _ask(graph, "dt-4", "What day is today?")
    traces = result.get("tool_results") or []
    assert any(t.get("tool") == "get_current_datetime" for t in traces)
    assert not any(t.get("tool") == "search_documents" for t in traces)
