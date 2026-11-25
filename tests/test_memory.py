"""E3: long-term memory - explicitly saved preferences, cross-thread use."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from core.models import Intent, IntentResult
from database.memory import PreferenceError
from tests.fakes import FakeGateway


def test_save_and_retrieve_preference(memory_repo):
    saved = memory_repo.save_preference(
        "preferred_response_style", "concise bullet points"
    )
    found = memory_repo.find("Preferred_Response_Style")  # case-insensitive
    assert found is not None
    assert found.value == "concise bullet points"
    assert found.id == saved.id


def test_saving_same_key_updates_value(memory_repo):
    memory_repo.save_preference("default_task_priority", "medium")
    updated = memory_repo.save_preference("default_task_priority", "high")
    assert updated.value == "high"
    assert len(memory_repo.list_preferences()) == 1


def test_delete_preference(memory_repo):
    saved = memory_repo.save_preference("preferred_task_format", "short titles")
    assert memory_repo.delete_preference(saved.id) is True
    assert memory_repo.list_preferences() == []
    assert memory_repo.delete_preference(saved.id) is False


def test_secret_like_keys_are_refused(memory_repo):
    for bad_key in ("openai_api_key", "my password", "auth token", "secret"):
        with pytest.raises(PreferenceError):
            memory_repo.save_preference(bad_key, "whatever")
    assert memory_repo.list_preferences() == []


def test_secret_like_values_are_refused(memory_repo):
    with pytest.raises(PreferenceError):
        memory_repo.save_preference(
            "note", "sk-ant-api03-abcdefghijklmnopqrstuv"
        )
    assert memory_repo.list_preferences() == []


def test_empty_and_oversized_preferences_refused(memory_repo):
    with pytest.raises(PreferenceError):
        memory_repo.save_preference("", "value")
    with pytest.raises(PreferenceError):
        memory_repo.save_preference("key", "   ")
    with pytest.raises(PreferenceError):
        memory_repo.save_preference("key", "v" * 501)


# ---------------------------------------------------------------------------
# Cross-thread availability through the graph
# ---------------------------------------------------------------------------
def _invoke(graph, thread_id: str, request: str):
    return graph.invoke(
        {
            "user_request": request,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=request)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def test_preference_saved_in_one_thread_available_in_new_conversation(
    graph, memory_repo, gateway
):
    """A preference saved (Settings page) before thread B starts is injected
    into thread B's prompts even though it was never mentioned there."""
    memory_repo.save_preference("explanation_level", "beginner-friendly")
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)

    _invoke(graph, "thread-b", "Explain how you handle my tasks")

    # The preference must appear in the messages the model received.
    injected = any(
        "explanation_level" in str(message.content)
        for messages, _ in gateway.seen_messages
        for message in messages
    )
    assert injected, "saved preference was not injected into the new thread"


def test_preferences_removed_are_not_injected(graph, memory_repo, gateway):
    saved = memory_repo.save_preference("preferred_task_format", "short titles")
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    _invoke(graph, "t-del-1", "hello")
    assert any(
        "preferred_task_format" in str(m.content)
        for messages, _ in gateway.seen_messages for m in messages
    )
    memory_repo.delete_preference(saved.id)
    gateway.seen_messages.clear()
    _invoke(graph, "t-del-2", "hello again")
    assert not any(
        "preferred_task_format" in str(m.content)
        for messages, _ in gateway.seen_messages for m in messages
    )


def test_task_proposal_prompt_receives_default_priority_preference(
    graph, memory_repo, gateway
):
    """The task proposal prompt must expose saved defaults to the model."""
    from core.models import TaskProposal

    memory_repo.save_preference("default_task_priority", "high")
    gateway.intent = IntentResult(intent=Intent.CREATE_TASK, confidence=0.9)
    gateway.proposal = TaskProposal(
        title="Read chapter 5", confirmation_message="I will create the task."
    )
    _invoke(graph, "t-pref-propose", "Add a task to read chapter 5")

    proposal_calls = [
        messages
        for messages, _ in gateway.seen_messages
        if any("default_task_priority" in str(m.content) for m in messages)
    ]
    assert proposal_calls, "preference not present in proposal prompt"
