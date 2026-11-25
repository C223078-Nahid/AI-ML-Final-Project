"""C1: prompt management and structured-output schemas."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from core.models import (
    Citation,
    GroundedAnswer,
    Intent,
    IntentResult,
    RetrievedChunk,
    TaskPriority,
    TaskProposal,
    UserPreference,
)
from core.prompts import (
    PROMPT_VERSIONS,
    format_history,
    grounded_answer_messages,
    intent_messages,
    preferences_block,
    refusal_text,
    render_passages,
    retrieval_planner_messages,
    task_proposal_messages,
    UNTRUSTED_BEGIN,
    UNTRUSTED_END,
)
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# IntentResult
# ---------------------------------------------------------------------------
def test_intent_result_valid_classifications():
    for intent in Intent:
        result = IntentResult(intent=intent, confidence=0.9)
        assert result.intent is intent


def test_intent_result_enforces_document_question_flags():
    result = IntentResult(
        intent=Intent.DOCUMENT_QUESTION, confidence=0.9,
        requires_retrieval=False, requires_tool=False,
    )
    # Deterministic consistency rules override the model's flags.
    assert result.requires_retrieval is True
    assert result.requires_tool is True


def test_intent_result_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=1.5)


def test_intent_result_rejects_invalid_intent():
    with pytest.raises(ValidationError):
        IntentResult(intent="download_the_internet", confidence=0.5)


def test_intent_result_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        IntentResult(confidence=0.5)  # intent missing


# ---------------------------------------------------------------------------
# TaskProposal
# ---------------------------------------------------------------------------
def _proposal(**overrides) -> dict:
    base = dict(
        title="Review lecture notes",
        description="Chapter 4",
        due_date=None,
        priority="high",
        confirmation_message="I will create the task 'Review lecture notes'.",
    )
    base.update(overrides)
    return base


def test_task_proposal_valid():
    proposal = TaskProposal(**_proposal())
    assert proposal.priority is TaskPriority.HIGH


def test_task_proposal_rejects_invalid_title():
    for bad_title in ("", "   ", "ab"):
        with pytest.raises(ValidationError):
            TaskProposal(**_proposal(title=bad_title))


def test_task_proposal_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        TaskProposal(**_proposal(priority="urgent"))


def test_task_proposal_rejects_missing_confirmation_message():
    payload = _proposal()
    del payload["confirmation_message"]
    with pytest.raises(ValidationError):
        TaskProposal(**payload)


def test_task_proposal_rejects_past_due_date():
    with pytest.raises(ValidationError):
        TaskProposal(**_proposal(due_date=date.today() - timedelta(days=1)))


def test_task_proposal_accepts_future_due_date():
    proposal = TaskProposal(**_proposal(due_date=date.today() + timedelta(days=3)))
    assert proposal.due_date is not None


def test_task_proposal_defaults_priority_to_medium():
    payload = _proposal()
    del payload["priority"]
    assert TaskProposal(**payload).priority is TaskPriority.MEDIUM


# ---------------------------------------------------------------------------
# GroundedAnswer / Citation
# ---------------------------------------------------------------------------
def test_grounded_answer_defaults():
    answer = GroundedAnswer(answer="Something")
    assert answer.citations == []
    assert answer.insufficient_evidence is False
    assert 0.0 <= answer.confidence <= 1.0


def test_grounded_answer_requires_answer_text():
    with pytest.raises(ValidationError):
        GroundedAnswer(answer="")


def test_citation_schema_round_trip():
    citation = Citation(
        document_id="d1", filename="plan.pdf", page_number=3,
        section="Tools", chunk_id="d1:0", supporting_passage="text",
    )
    data = citation.model_dump(mode="json")
    assert Citation(**data) == citation


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def test_all_prompts_are_versioned():
    for key in (
        "system_assistant",
        "intent_classification",
        "grounded_answer",
        "task_proposal",
        "retrieval_planner",
        "safety_refusal",
    ):
        assert key in PROMPT_VERSIONS
        assert PROMPT_VERSIONS[key]


def test_intent_prompt_lists_every_supported_intent():
    messages = intent_messages("hello", [])
    prompt_text = messages[0].content + str(messages[1].content)
    for intent in Intent:
        assert intent.value in prompt_text


def test_grounded_answer_prompt_wraps_content_as_untrusted():
    chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1", filename="plan.pdf",
            page_number=3, section="Tools", text="Qdrant was selected.",
        )
    ]
    prompt_text = grounded_answer_messages("What DB?", chunks, None)[0].content
    assert UNTRUSTED_BEGIN in prompt_text
    assert UNTRUSTED_END in prompt_text
    assert 'source_id=S1' in prompt_text  # short ordinal ids (reliable to echo)
    assert 'file="plan.pdf"' in prompt_text
    # The prompt must instruct the model to treat content as data.
    assert "UNTRUSTED" in prompt_text
    assert "ignore" in prompt_text.lower()
    # Personal facts must be answered from the passages, not abstained on.
    assert "personal facts" in prompt_text


def test_preferences_block_renders_saved_preferences():
    prefs = [
        UserPreference(
            id="m1", key="default_task_priority", value="high",
            created_at="2026-01-01", updated_at="2026-01-01",
        )
    ]
    block = preferences_block(prefs)
    assert "default_task_priority" in block and "high" in block
    assert "not saved any" in preferences_block(None)


def test_task_proposal_prompt_includes_today_and_preferences():
    prefs = [
        UserPreference(
            id="m1", key="preferred_task_format", value="short titles",
            created_at="2026-01-01", updated_at="2026-01-01",
        )
    ]
    today = date(2026, 11, 21)
    prompt_text = task_proposal_messages("add a task", [], today, prefs)[0].content
    assert "2026-11-21" in prompt_text
    assert "preferred_task_format" in prompt_text


def test_refusal_text_includes_reason_and_alternative():
    text = refusal_text("Not allowed.", "Try searching documents instead.")
    assert "Not allowed." in text
    assert "Try searching documents instead." in text
    bare = refusal_text("Not allowed.")
    assert "Not allowed." in bare


def test_format_history_renders_roles():
    history = [
        HumanMessage(content="List my open tasks."),
        AIMessage(content="You have two open tasks."),
        HumanMessage(content="What is its due date?"),
    ]
    rendered = format_history(history)
    assert "Human: List my open tasks." in rendered
    assert "AI: You have two open tasks." in rendered


def test_retrieval_planner_prompt_bounds_tool_calls():
    text = retrieval_planner_messages("Which database?", max_calls=3)[0].content
    assert "at most 3" in text


def test_render_passages_marks_all_chunks_untrusted():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", document_id="d1", filename="f.pdf",
            page_number=i, section=None, text=f"passage {i}",
        )
        for i in range(3)
    ]
    rendered = render_passages(chunks)
    assert rendered.count(UNTRUSTED_BEGIN) == 1
    # Ordinal source ids S1..S3, in the order the chunks were provided.
    for i in range(1, 4):
        assert f"source_id=S{i}" in rendered
    assert "source_id=c0" not in rendered
