"""C7: LangGraph workflow branches, conversation memory, and checkpointing."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from core.llm_gateway import ModelGatewayError, ErrorCategory
from core.models import (
    Citation,
    GroundedAnswer,
    Intent,
    IntentResult,
    TaskPriority,
    TaskProposal,
)
from tests.fakes import FakeGateway, tool_call_message

CHUNK_TEXT = "Qdrant was selected as the vector database for the project."


@pytest.fixture
def indexed_doc(ingestion, doc_repo):
    result = ingestion.ingest_file("plan.txt", CHUNK_TEXT.encode())
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    return document, f"{document.document_id}:0"


def _invoke(graph, thread_id: str, request: str):
    return graph.invoke(
        {
            "user_request": request,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=request)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


# ---------------------------------------------------------------------------
# Document question branch
# ---------------------------------------------------------------------------
def test_document_reference_overrides_general_classification(
    graph, gateway, indexed_doc
):
    """RAG connectivity fix: a misrouted document question still reaches RAG.

    The classifier (free endpoint, prompt-mode structured output) labels
    "Summarize my uploaded document" as general_conversation. The
    deterministic document-reference override must send it to the document
    branch so retrieval actually runs and the retrieved content reaches the
    model, instead of a generic "I cannot access your documents" reply.
    """
    document, chunk_id = indexed_doc
    gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.8
    )
    gateway.agent_script = [
        tool_call_message("search_documents", {"query": CHUNK_TEXT}),
    ]
    gateway.grounded = GroundedAnswer(
        answer="Your document states that Qdrant was selected.",
        citations=[
            Citation(
                document_id=document.document_id, filename="plan.txt",
                chunk_id=chunk_id, supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )
    result = _invoke(
        graph, "t-override", "Summarize what my uploaded document says."
    )
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["citations"][0]["chunk_id"] == chunk_id
    # Retrieved content reached the model's answer-generation input.
    grounded_prompt_texts = [
        str(getattr(m, "content", "")) for m in gateway.seen_messages[-1][0]
    ]
    assert any(CHUNK_TEXT in text for text in grounded_prompt_texts)


def test_document_reference_override_requires_indexed_documents(
    graph, gateway, settings
):
    """Without any indexed documents the override must NOT fire.

    The user saying "summarize my document" with an empty library should
    keep the general route and get an honest answer, not a fake RAG answer.
    """
    gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.8
    )
    result = _invoke(
        graph, "t-override-empty", "Summarize what my uploaded document says."
    )
    payload = result["final_response"]
    assert payload["kind"] == "conversation"  # stayed on the general branch


def test_general_question_without_document_reference_not_hijacked(
    graph, gateway, indexed_doc
):
    """A general question must stay general even when documents exist."""
    gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    result = _invoke(
        graph, "t-general-keep", "What is the capital of France?"
    )
    payload = result["final_response"]
    assert payload["kind"] == "conversation"


def test_document_question_branch_with_agent_tool_loop(graph, gateway, indexed_doc):
    document, chunk_id = indexed_doc
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95)
    gateway.agent_script = [
        tool_call_message("search_documents", {"query": CHUNK_TEXT}),
    ]
    gateway.grounded = GroundedAnswer(
        answer="Qdrant was selected as the vector database.",
        citations=[
            Citation(
                document_id=document.document_id, filename="plan.txt",
                page_number=None, section=None, chunk_id=chunk_id,
                supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )
    # The question shares no keywords with the chunk, so hybrid direct
    # retrieval finds nothing and the bounded planner loop (tool-calling)
    # is what fetches the passage - the multi-hop fallback path.
    result = _invoke(graph, "t-doc", "Which option did I pick in the end?")

    assert gateway.chat_calls_with_tools >= 1  # agent loop used the tool
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["citations"][0]["chunk_id"] == chunk_id
    assert result["tool_results"][0]["tool"] == "search_documents"
    # The assistant message carries the payload for the UI.
    last_message = result["messages"][-1]
    assert last_message.additional_kwargs["pkta_payload"]["kind"] == "answer"


def test_document_question_falls_back_to_direct_retrieval(graph, gateway, indexed_doc):
    """If the model never calls the tool, the node still retrieves directly."""
    document, chunk_id = indexed_doc
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9)
    gateway.agent_script = []  # planner replies DONE without tool calls
    gateway.grounded = GroundedAnswer(
        answer="According to your plan, Qdrant was selected.",
        citations=[
            Citation(
                document_id=document.document_id, filename="plan.txt",
                chunk_id=chunk_id, supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )
    result = _invoke(graph, "t-direct", CHUNK_TEXT)  # exact match query
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["citations"][0]["chunk_id"] == chunk_id


def test_abstention_when_no_evidence(graph, gateway, indexed_doc):
    """No matching content in a NON-empty library -> honest abstention.

    The document is indexed, but the question is unrelated to it, retrieval
    returns nothing above the score threshold, and the assistant must
    abstain instead of inventing an answer (task section 6, "insufficient
    evidence").
    """
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9)
    gateway.agent_script = []
    gateway.grounded = GroundedAnswer(
        answer="Unsupportable claim.", citations=[], confidence=0.9
    )
    result = _invoke(
        graph, "t-abstain", "What is the airspeed velocity of an unladen swallow?"
    )
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["insufficient_evidence"] is True
    assert "could not find" in payload["text"]
    assert payload.get("library_empty") is not True  # library was NOT empty


def test_empty_library_gets_distinct_message(graph, gateway, settings):
    """Asking a document question with NO indexed documents.

    This is an application-state condition, not missing evidence: the
    response must say that nothing is indexed (and how to fix it) rather
    than claiming the documents lack the answer.
    """
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9)
    gateway.agent_script = []
    result = _invoke(
        graph, "t-empty-lib", "What does my uploaded document say about databases?"
    )
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["insufficient_evidence"] is True
    assert payload["library_empty"] is True
    assert "no documents are currently indexed" in payload["text"]
    # The grounded-answer model was never called: there was nothing to ground.
    assert "GroundedAnswer" not in gateway.structured_calls


def test_hallucinated_citation_dropped_in_graph_flow(graph, gateway, indexed_doc):
    """A citation that references nothing real is rejected, never shown.

    New contract (follow-up task, section 4: "reject or correct citation"):
    the invalid CITATION is dropped and the model gets one corrective retry;
    when the retry still cites nothing real, the answer text is kept but
    flagged ``unverified`` with NO citations - the user never sees a
    fabricated source, and a possibly-correct answer is not silently
    discarded either.
    """
    document, chunk_id = indexed_doc
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9)
    gateway.agent_script = [tool_call_message("search_documents", {"query": CHUNK_TEXT})]
    gateway.grounded = GroundedAnswer(
        answer="Some answer.",
        citations=[
            Citation(document_id="ghost", filename="ghost.pdf",
                     chunk_id="ghost:0", supporting_passage="invented"),
        ],
        confidence=0.9,
    )
    result = _invoke(graph, "t-halluc", "Which database?")
    payload = result["final_response"]
    # The fabricated citation is rejected in BOTH attempts.
    assert payload["citations"] == []
    assert payload.get("unverified") is True
    # ...and the model twice refused to cite anything real, so the answer is
    # marked unverified rather than presented as a grounded, cited answer.
    assert payload["insufficient_evidence"] is False
    assert payload["text"] == "Some answer."
    # The corrective retry really happened (two grounded calls: first + retry).
    assert gateway.structured_calls.count("GroundedAnswer") == 2


# ---------------------------------------------------------------------------
# Task listing branch
# ---------------------------------------------------------------------------
def test_task_listing_branch(graph, gateway, task_repo):
    task_repo.create_task(
        TaskProposal(title="Finish ML assignment", confirmation_message="ok"),
        "op-list-1",
    )
    gateway.intent = IntentResult(intent=Intent.LIST_TASKS, confidence=0.9)
    result = _invoke(graph, "t-list", "Show my open tasks")
    payload = result["final_response"]
    assert payload["kind"] == "task_list"
    assert any(t["title"] == "Finish ML assignment" for t in payload["tasks"])
    assert result["tool_results"][0]["tool"] == "list_tasks"


# ---------------------------------------------------------------------------
# General conversation branch (with read-only tool loop)
# ---------------------------------------------------------------------------
def test_general_conversation_branch_with_tools(graph, gateway, task_repo):
    task_repo.create_task(
        TaskProposal(title="Buy milk", confirmation_message="ok"), "op-gen-1"
    )
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    gateway.agent_script = [
        tool_call_message("list_tasks", {"status": "all"}, call_id="c2"),
        AIMessage(content="Your task 'Buy milk' is still open."),
    ]
    result = _invoke(graph, "t-gen", "What is my second task about?")
    payload = result["final_response"]
    assert payload["kind"] == "conversation"
    assert "Buy milk" in payload["text"]
    assert result["tool_results"][0]["tool"] == "list_tasks"


def test_general_conversation_simple(graph, gateway):
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    gateway.agent_script = [AIMessage(content="Hello! How can I help?")]
    result = _invoke(graph, "t-hello", "Hi there")
    assert result["final_response"]["text"] == "Hello! How can I help?"


# ---------------------------------------------------------------------------
# Unsupported requests and guardrails
# ---------------------------------------------------------------------------
def test_unsupported_request_refused_without_tool_work(graph, gateway):
    gateway.intent = IntentResult(intent=Intent.UNSUPPORTED_REQUEST, confidence=0.95)
    # Calendar sync is unsupported but does NOT match any guardrail pattern,
    # so it reaches the intent router and is refused there.
    result = _invoke(graph, "t-unsup", "Sync my tasks with Google Calendar")
    payload = result["final_response"]
    assert payload["kind"] == "refusal"
    assert "outside the operations" in payload["text"]
    # Only the intent classification ran; no answer or tool work happened.
    assert gateway.structured_calls == ["IntentResult"]


def test_guardrail_blocks_banned_request_before_any_model_call(graph, gateway, settings):
    result = _invoke(
        graph, "t-banned", "Please browse the web and find me a cheap hotel"
    )
    payload = result["final_response"]
    assert payload["kind"] == "refusal"
    assert gateway.structured_calls == []  # no model was called at all
    assert gateway.seen_messages == []


def test_too_long_message_rejected(graph, gateway, settings):
    result = _invoke(graph, "t-long", "x" * (settings.max_message_chars + 1))
    payload = result["final_response"]
    assert payload["kind"] == "error"
    assert "too long" in payload["text"] or "too long" in payload.get("error", "")


# ---------------------------------------------------------------------------
# Error branch
# ---------------------------------------------------------------------------
def test_model_failure_produces_error_response(graph, gateway):
    # Total provider outage: both the structured classifier and the chat
    # model are down. The request must still end in a clean error response.
    gateway.structured_error = ModelGatewayError(
        "all providers failed", ErrorCategory.UNKNOWN
    )
    gateway.chat_error = ModelGatewayError(
        "all providers failed", ErrorCategory.UNKNOWN
    )
    result = _invoke(graph, "t-err", "Hello")
    payload = result["final_response"]
    assert payload["kind"] == "error"
    assert "went wrong" in payload["text"]


def test_intent_failure_falls_back_to_heuristic_classification(
    graph, gateway, indexed_doc
):
    """Classifier outage alone must not kill document questions.

    The intent call fails, but the deterministic heuristic classifier still
    routes the request to the document branch, retrieval runs, and the
    retrieved content reaches the grounded-answer prompt (RAG connectivity).
    """
    document, chunk_id = indexed_doc
    gateway.intent_error = ModelGatewayError(
        "classifier unavailable", ErrorCategory.RATE_LIMIT
    )
    gateway.agent_script = [
        tool_call_message("search_documents", {"query": CHUNK_TEXT}),
    ]
    gateway.grounded = GroundedAnswer(
        answer="According to your plan, Qdrant was selected.",
        citations=[
            Citation(
                document_id=document.document_id, filename="plan.txt",
                chunk_id=chunk_id, supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )
    result = _invoke(
        graph, "t-heur", "What database does my document talk about?"
    )
    # The classifier was attempted (and failed)...
    assert gateway.structured_calls[0] == "IntentResult"
    # ...yet the request still completed through the document branch.
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["citations"][0]["chunk_id"] == chunk_id
    assert "GroundedAnswer" in gateway.structured_calls
    # And the retrieved chunk text actually reached the answer prompt.
    grounded_messages = gateway.seen_messages[-1][0]
    assert any(
        CHUNK_TEXT in str(getattr(m, "content", "")) for m in grounded_messages
    )


# ---------------------------------------------------------------------------
# Conversation memory (C4)
# ---------------------------------------------------------------------------
def test_conversation_memory_preserves_context_within_thread(graph_factory, gateway):
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    graph = graph_factory()
    config = {"configurable": {"thread_id": "t-mem"}}
    graph.invoke(
        {"user_request": "My favorite course is machine learning.",
         "thread_id": "t-mem",
         "messages": [HumanMessage("My favorite course is machine learning.")]},
        config,
    )
    graph.invoke(
        {"user_request": "What is my favorite course?",
         "thread_id": "t-mem",
         "messages": [HumanMessage("What is my favorite course?")]},
        config,
    )
    state = graph.get_state(config)
    messages = state.values["messages"]
    assert len(messages) == 4  # human/ai/human/ai
    # The second turn's model call saw the first turn in its history.
    second_call_messages = gateway.seen_messages[-1][0]
    history_text = " ".join(str(m.content) for m in second_call_messages)
    assert "machine learning" in history_text


def test_threads_are_isolated(graph_factory, gateway):
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    graph = graph_factory()
    _invoke(graph, "t-a", "Secret message for thread A")
    _invoke(graph, "t-b", "Hello thread B")
    state_a = graph.get_state({"configurable": {"thread_id": "t-a"}})
    state_b = graph.get_state({"configurable": {"thread_id": "t-b"}})
    texts_a = [str(m.content) for m in state_a.values["messages"]]
    texts_b = [str(m.content) for m in state_b.values["messages"]]
    assert any("thread A" in t for t in texts_a)
    assert not any("thread A" in t for t in texts_b)


def test_checkpoint_state_recovered_by_new_graph_instance(graph_factory, gateway):
    """Simulates an application restart: same checkpointer file, new graph."""
    gateway.intent = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    graph = graph_factory()
    _invoke(graph, "t-ckpt", "Remember this conversation")
    restarted = graph_factory()  # fresh instance, same checkpoint database
    snapshot = restarted.get_state({"configurable": {"thread_id": "t-ckpt"}})
    texts = [str(m.content) for m in snapshot.values["messages"]]
    assert any("Remember this conversation" in t for t in texts)
    assert snapshot.next == ()  # run completed; nothing pending
