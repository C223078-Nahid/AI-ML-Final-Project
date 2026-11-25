"""Acceptance tests for the follow-up task (sections 2-10): critical RAG
fix, query routing, and UI/processing-state contracts.

Synthetic test document and values come directly from the task (section
2.1); no real personal information is used anywhere.

- Test A: basic personal-information retrieval (name / DOB / favourites)
- Test B: the ACTUAL model input contains the retrieved facts (real
  ModelGateway, prompt-mode structured output, scripted provider model)
- Test C: evidence validation (phone number is genuinely absent)
- Test D: general-knowledge routing (no document abstention)
- Test E: current date/time through MODEL-DRIVEN tool calling (follow-up
  task 4, section 7: the model invokes get_current_datetime; the
  application clock in APP_TIMEZONE remains the data source)
- Test F: mixed conversation - the router switches paths within one thread
- Test G: shared document library across chats, isolated histories
- Test H: UI canaries (sidebar order, composer layout, processing state)
- Regression repros from the Phase-1 investigation, now fixed
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import HumanMessage

from core.llm_gateway import ModelGateway
from core.models import Citation, GroundedAnswer, Intent, IntentResult
from tests.fakes import FakeGateway

TESTING_TXT = (
    b"Personal Informations:\n\n"
    b"Name: Alex Example\n\n"
    b"Date of Birth: 12-03-1995\n\n"
    b"Favourite: Watching anime, Reading manga, Eating cake.\n"
)

VERBATIM_CHUNK = TESTING_TXT.decode().strip()

NAME_QUESTION = "What is my name? Hint: can be found in the TXT file."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def testing_doc(ingestion, doc_repo):
    """The exact synthetic document from the task, indexed once."""
    result = ingestion.ingest_file("Testing.txt", TESTING_TXT)
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    return document, f"{document.document_id}:0"


class _StatefulGateway(FakeGateway):
    """Gateway that replays scripted intents (and optional grounded answers)
    per turn (Tests D/E/F)."""

    def __init__(self, intents, grounded=None, chat_text="general answer",
                 grounded_sequence=None):
        super().__init__()
        self._intents = list(intents)
        self._grounded_sequence = list(grounded_sequence or [])
        self._calls = 0
        self._grounded_calls = 0
        self.grounded = grounded
        self.chat_text = chat_text

    def structured(self, schema, messages, run_name=None, metadata=None):
        if schema.__name__ == "IntentResult":
            intent = self._intents[min(self._calls, len(self._intents) - 1)]
            self._calls += 1
            self.intent = intent
        elif schema.__name__ == "GroundedAnswer" and self._grounded_sequence:
            index = min(self._grounded_calls, len(self._grounded_sequence) - 1)
            self._grounded_calls += 1
            self.grounded = self._grounded_sequence[index]
        return super().structured(schema, messages, run_name, metadata)


def _grounded_for(document_id: str, answer: str, passage: str,
                  chunk_id: str = "S1") -> GroundedAnswer:
    """A model-shaped grounded answer citing the first source by ordinal."""
    return GroundedAnswer(
        answer=answer,
        citations=[Citation(
            document_id="whatever-the-model-put",  # the model does not know it
            filename="Testing.txt",
            chunk_id=chunk_id,                     # short ordinal source id
            supporting_passage=passage,
        )],
        confidence=0.9,
    )


def _ask(graph, thread_id, question):
    return graph.invoke(
        {
            "user_request": question,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=question)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


# ---------------------------------------------------------------------------
# Test A: basic personal-information retrieval
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "expected_fragment"),
    [
        (NAME_QUESTION, "Alex Example"),
        ("What is my date of birth?", "12-03-1995"),
        ("What are my favourite activities?", "Watching anime"),
    ],
)
def test_a_personal_info_retrieval(services, settings, testing_doc, question,
                                   expected_fragment):
    """Name / DOB / favourites are answered from Testing.txt with citations.

    The retrieval is hybrid: even though the question shares no exact text
    with the chunk (fake embeddings score it below the threshold), the
    FTS5 lexical path finds it through the keyword match.
    """
    document, chunk_id = testing_doc
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
        grounded=_grounded_for(
            document.document_id,
            f"According to Testing.txt, the answer involves {expected_fragment}.",
            "Name: Alex Example",
        ),
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-a", question)

    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["insufficient_evidence"] is False
    assert expected_fragment in payload["text"]
    # The citation points at the REAL chunk (metadata corrected by the app).
    citations = payload["citations"]
    assert citations and citations[0]["filename"] == "Testing.txt"
    assert citations[0]["chunk_id"] == chunk_id
    assert citations[0]["document_id"] == document.document_id


def test_a_single_chunk_preserves_complete_text(testing_doc, retriever):
    """One chunk is acceptable for a small file - but it must be complete.

    The verbatim chunk keeps every label and paragraph break, and the UI
    chunk count matches the actual stored points.
    """
    document, chunk_id = testing_doc
    assert document.chunk_count == 1
    hits = retriever.search("name birth favourite", top_k=3)
    assert hits and hits[0].chunk_id == chunk_id
    text = hits[0].text
    assert "Name: Alex Example" in text
    assert "Date of Birth: 12-03-1995" in text
    assert "Favourite: Watching anime, Reading manga, Eating cake." in text
    assert "\n\n" in text  # paragraph boundaries preserved


# ---------------------------------------------------------------------------
# Test B: the ACTUAL model input contains the facts
# ---------------------------------------------------------------------------
def _grounded_json(document_id: str) -> str:
    return json.dumps(
        {
            "answer": "Your name is Alex Example.",
            "citations": [
                {
                    "document_id": document_id,
                    "filename": "Testing.txt",
                    "page_number": None,
                    "section": None,
                    "chunk_id": "S1",
                    "supporting_passage": "Name: Alex Example",
                }
            ],
            "confidence": 0.9,
            "insufficient_evidence": False,
        }
    )


def test_b_actual_model_input_contains_retrieved_facts(
    services, settings, testing_doc, retriever
):
    """The strongest possible plumbing proof: a REAL ModelGateway in the
    OpenRouter prompt-mode configuration, with the provider model replaced
    by a scripted text model that records exactly what the gateway sends.

    Asserts (task section 2.3):
    - the fully formatted final messages contain the document facts,
    - the user's question is present,
    - source identifiers and metadata are preserved,
    - the structured output is parsed back into the answer the user sees.
    """
    from tests.fakes import SequentialTextModel

    document, chunk_id = testing_doc
    provider_model = SequentialTextModel(
        [
            # 1st structured call: intent classification
            json.dumps(
                {"intent": "document_question", "confidence": 0.95,
                 "requires_retrieval": True, "requires_tool": True,
                 "clarification_needed": False}
            ),
            # 2nd structured call: grounded answer
            _grounded_json(document.document_id),
        ]
    )
    gateway = ModelGateway(
        primary=provider_model,
        fallback=None,
        primary_name="nvidia/nemotron-3-ultra-550b-a55b:free",
        fallback_name="claude-haiku-5-5",
        primary_provider="openrouter",
        fallback_provider="anthropic",
        primary_structured_mode="prompt",
        fallback_structured_mode="native",
    )
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-b", NAME_QUESTION)
    payload = result["final_response"]

    # The provider model received exactly two structured calls.
    assert provider_model.calls == 2
    grounded_payload = provider_model.received[1]
    rendered = "\n".join(str(getattr(m, "content", "")) for m in grounded_payload)
    # The retrieved document facts reached the actual model request.
    assert "Name: Alex Example" in rendered
    assert "Date of Birth: 12-03-1995" in rendered
    assert "Watching anime, Reading manga, Eating cake." in rendered
    # The user's question is present.
    assert NAME_QUESTION in rendered
    # Source identifiers and metadata are preserved in the passage headers.
    assert "source_id=S1" in rendered
    assert 'file="Testing.txt"' in rendered
    # The JSON contract (prompt-mode structured output) is appended.
    assert "OUTPUT FORMAT REQUIREMENTS" in rendered
    # The parsed structured answer is what the user sees.
    assert payload["text"] == "Your name is Alex Example."
    assert payload["citations"][0]["chunk_id"] == chunk_id


def test_b_dev_capture_hook_records_final_messages(
    services, settings, testing_doc, tmp_path, monkeypatch
):
    """The PKTA_DEV_CAPTURE_DIR verification mechanism (task 2.3): the
    final message list is captured immediately before the model call, and
    the captured record contains the retrieved facts."""
    document, _ = testing_doc
    monkeypatch.setenv("PKTA_DEV_CAPTURE_DIR", str(tmp_path))

    from tests.fakes import SequentialTextModel

    provider_model = SequentialTextModel(
        [
            json.dumps({"intent": "document_question", "confidence": 0.95}),
            _grounded_json(document.document_id),
        ]
    )
    gateway = ModelGateway(
        primary=provider_model, fallback=None,
        primary_name="nemotron-free", fallback_name="haiku",
        primary_provider="openrouter", fallback_provider="anthropic",
        primary_structured_mode="prompt", fallback_structured_mode="native",
    )
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    _ask(graph, "t-b2", NAME_QUESTION)

    capture_file = tmp_path / "model_messages.jsonl"
    assert capture_file.exists()
    records = [json.loads(line) for line in capture_file.read_text().splitlines()]
    grounded_records = [r for r in records if r["kind"] == "structured"
                        and r["run_name"] == "GroundedAnswer"]
    assert grounded_records, records
    content = "\n".join(m["content"] for m in grounded_records[-1]["messages"])
    assert "Name: Alex Example" in content
    assert NAME_QUESTION in content


# ---------------------------------------------------------------------------
# Test C: evidence validation - genuinely missing information
# ---------------------------------------------------------------------------
def test_c_missing_information_abstains_honestly(services, settings, testing_doc):
    """'What is my phone number?' is absent from the document: the model
    receives the retrieved personal-information context, cannot find a
    phone number in it, and abstains honestly - never invents a number."""
    document, _ = testing_doc
    abstaining = GroundedAnswer(
        answer="I could not find sufficient information about this in the "
               "documents currently indexed.",
        citations=[], confidence=0.4, insufficient_evidence=True,
    )
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
        grounded=abstaining,
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    # The question mentions "personal informations", so retrieval DOES
    # return the chunk - the abstention is the model judging the EVIDENCE
    # (no phone number in it), not a retrieval miss.
    result = _ask(
        graph, "t-c",
        "What is my phone number? It should be in my personal informations.",
    )
    payload = result["final_response"]
    assert payload["insufficient_evidence"] is True
    assert payload["citations"] == []
    assert "could not find" in payload["text"]
    # The model really received the retrieved context and abstained twice
    # (first attempt + corrective retry).
    assert services.gateway.structured_calls.count("GroundedAnswer") == 2
    seen = "\n".join(
        str(getattr(m, "content", ""))
        for messages, _ in services.gateway.seen_messages
        for m in messages
    )
    assert "Name: Alex Example" in seen  # context reached the model
    # No phone number was invented anywhere in the response.
    assert not any(ch.isdigit() for ch in payload["text"])


# ---------------------------------------------------------------------------
# Test D: general knowledge routing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "What is LangGraph?",
    "What is the difference between SQL and NoSQL?",
])
def test_d_general_knowledge_answers_without_rag(services, settings,
                                                 testing_doc, question):
    """General-knowledge questions use the model directly - no retrieval,
    no document abstention."""
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.GENERAL_KNOWLEDGE, confidence=0.9),
        chat_text="LangGraph is a framework for building stateful "
                  "multi-actor applications with LLMs.",
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-d", question)
    payload = result["final_response"]
    assert payload["kind"] == "conversation"
    assert payload.get("insufficient_evidence") is not True
    # No retrieval was executed for a general-knowledge question.
    assert not any(t.get("tool") == "search_documents"
                   for t in result.get("tool_results") or [])


def test_d_general_knowledge_not_hijacked_in_document_thread(
    services, settings, testing_doc
):
    """The Phase-1 R4 repro, now fixed: 'What is LangGraph?' inside a thread
    that already answered document questions stays a general question (the
    old <=4-word follow-up heuristic hijacked it into RAG)."""
    document, _ = testing_doc
    gateway = _StatefulGateway(
        intents=[
            IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
            IntentResult(intent=Intent.GENERAL_KNOWLEDGE, confidence=0.95),
        ],
        grounded=_grounded_for(document.document_id, "Your name is Alex Example.",
                               "Name: Alex Example"),
        chat_text="LangGraph is an orchestration framework.",
    )
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    _ask(graph, "t-d2", NAME_QUESTION)               # document mode entered
    result = _ask(graph, "t-d2", "What is LangGraph?")  # general knowledge
    payload = result["final_response"]
    assert payload["kind"] == "conversation"
    assert not any(t.get("tool") == "search_documents"
                   for t in result.get("tool_results") or [])


# ---------------------------------------------------------------------------
# Test E: current date/time routing
# ---------------------------------------------------------------------------
_FROZEN_NOW = datetime(2026, 11, 24, 9, 41, 0, tzinfo=ZoneInfo("Asia/Dhaka"))


def test_e_current_date_uses_application_clock(services, settings, testing_doc,
                                               monkeypatch):
    """'What day is today?' - the MODEL decides to call the registered
    get_current_datetime tool; the tool returns the application clock in
    the configured APP_TIMEZONE; the final answer uses it.  Never from
    documents, never from model training data (follow-up task 4, sec. 7)."""
    import core.datetime_util as dt_util
    from langchain_core.messages import AIMessage
    from tests.fakes import tool_call_message

    monkeypatch.setattr(dt_util, "now", lambda tz=None: _FROZEN_NOW)
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.CURRENT_DATETIME, confidence=0.9),
        agent_script=[
            tool_call_message("get_current_datetime", {}),
            AIMessage(content=(
                "Today is Tuesday, 24 November 2026. The current time is "
                "09:41 (BST, Asia/Dhaka)."
            )),
        ],
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-e", "What day is today?")
    payload = result["final_response"]

    # 1. The MODEL decided to call the tool (no deterministic node).
    assert payload["kind"] == "conversation"
    traces = result.get("tool_results") or []
    datetime_trace = [t for t in traces if t.get("tool") == "get_current_datetime"]
    assert len(datetime_trace) == 1
    # 2. The tool returned the ACTUAL application-clock payload (frozen
    #    clock, configured timezone) - verified against Python's own
    #    timezone-aware datetime, independently of the model's words.
    tool_result = datetime_trace[0]["result"]
    assert "24 November 2026" in tool_result
    assert "Tuesday" in tool_result
    assert "Asia/Dhaka" in tool_result
    assert "2026-11-24T09:41:00" in tool_result
    # 3. The final answer is built on the tool output.
    assert "24 November 2026" in payload["text"]
    # 4. No document retrieval ran for a current-date question.
    assert not any(t.get("tool") == "search_documents" for t in traces)
    # 5. No extra model call happened (intent + agent turn + final turn).
    assert services.gateway.structured_calls == ["IntentResult"]


def test_e_datetime_safety_net_overrides_misrouting(services, settings,
                                                    testing_doc):
    """Even when the classifier mislabels a date question as
    document_question (the original bug), the datetime safety net reroutes
    it to the tool-calling agent - never into document retrieval - and the
    model answers through get_current_datetime (follow-up task 4, sec. 7)."""
    from langchain_core.messages import AIMessage
    from tests.fakes import tool_call_message

    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9),
        agent_script=[
            tool_call_message("get_current_datetime", {}),
            AIMessage(content="Today is Tuesday, 24 November 2026 (Asia/Dhaka)."),
        ],
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-e2", "What day is today?")
    payload = result["final_response"]
    assert result["intent"]["intent"] == "current_datetime"
    assert payload["kind"] == "conversation"
    traces = result.get("tool_results") or []
    assert any(t.get("tool") == "get_current_datetime" for t in traces)
    # The request never reached the RAG branch.
    assert not any(t.get("tool") == "search_documents" for t in traces)


def test_e_datetime_in_document_thread_stays_datetime(services, settings,
                                                      testing_doc, monkeypatch):
    """The Phase-1 R4 repro (exact user report), now fixed: after a document
    answer, 'What day is today?' is NOT hijacked into document abstention.
    The datetime net sends it to the tool-calling agent (follow-up task 4,
    section 7), where the model calls get_current_datetime."""
    import core.datetime_util as dt_util
    from langchain_core.messages import AIMessage
    from tests.fakes import tool_call_message

    monkeypatch.setattr(dt_util, "now", lambda tz=None: _FROZEN_NOW)
    document, _ = testing_doc

    class _DocThenDatetimeGateway(_StatefulGateway):
        """First turn: document answer.  Second turn (a chat call with
        tools bound, i.e. the agent loop): call the datetime tool, then
        answer."""

        def __init__(self, grounded):
            super().__init__(
                intents=[IntentResult(intent=Intent.DOCUMENT_QUESTION,
                                      confidence=0.95)],
                grounded=grounded,
            )
            self.agent_script = [
                tool_call_message("get_current_datetime", {}),
                AIMessage(content=(
                    "Today is Tuesday, 24 November 2026. The current time "
                    "is 09:41 (BST, Asia/Dhaka)."
                )),
            ]

    services.gateway = _DocThenDatetimeGateway(
        _grounded_for(document.document_id, "Your name is Alex Example.",
                      "Name: Alex Example")
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    first = _ask(graph, "t-e3", NAME_QUESTION)
    assert first["final_response"]["kind"] == "answer"
    second = _ask(graph, "t-e3", "What day is today?")
    payload = second["final_response"]
    assert payload["kind"] == "conversation"
    assert "Tuesday" in payload["text"]
    assert "24 November 2026" in payload["text"]
    traces = second.get("tool_results") or []
    assert any(t.get("tool") == "get_current_datetime" for t in traces)
    assert not any(t.get("tool") == "search_documents" for t in traces)


# ---------------------------------------------------------------------------
# Test F: mixed conversation - routing switches inside one thread
# ---------------------------------------------------------------------------
def test_f_mixed_conversation_switches_paths(services, settings, testing_doc,
                                             monkeypatch):
    """The exact sequence from the task: document -> general -> datetime ->
    document, in one chat, with correct switching."""
    import core.datetime_util as dt_util

    monkeypatch.setattr(dt_util, "now", lambda tz=None: _FROZEN_NOW)
    document, _ = testing_doc
    from langchain_core.messages import AIMessage
    from tests.fakes import tool_call_message

    gateway = _StatefulGateway(
        intents=[
            IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
            IntentResult(intent=Intent.GENERAL_KNOWLEDGE, confidence=0.9),
            IntentResult(intent=Intent.CURRENT_DATETIME, confidence=0.9),
            IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
        ],
        grounded_sequence=[
            _grounded_for(document.document_id,
                          "Your name is Alex Example.",
                          "Name: Alex Example"),
            _grounded_for(document.document_id,
                          "Your date of birth is 12-03-1995.",
                          "Date of Birth: 12-03-1995"),
        ],
        chat_text="LangGraph routes work through a state graph.",
    )
    # Turn 2 (general knowledge) and turn 3 (datetime) both run in the
    # agent loop: script tool calls for each, then final answers.
    gateway.agent_script = [
        AIMessage(content="LangGraph routes work through a state graph."),
        tool_call_message("get_current_datetime", {}),
        AIMessage(content="Today is Tuesday, 24 November 2026 (Asia/Dhaka)."),
    ]
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    thread = "t-f"

    one = _ask(graph, thread, NAME_QUESTION)
    assert one["final_response"]["kind"] == "answer"
    assert "Alex Example" in one["final_response"]["text"]
    assert one["final_response"]["citations"]

    two = _ask(graph, thread, "What is LangGraph?")
    assert two["final_response"]["kind"] == "conversation"
    assert not two["final_response"].get("insufficient_evidence")

    three = _ask(graph, thread, "What day is today?")
    assert three["final_response"]["kind"] == "conversation"
    assert "24 November 2026" in three["final_response"]["text"]
    assert any(t.get("tool") == "get_current_datetime"
               for t in three.get("tool_results") or [])

    four = _ask(graph, thread, "What is my date of birth?")
    assert four["final_response"]["kind"] == "answer"
    assert "12-03-1995" in four["final_response"]["text"]
    assert four["final_response"]["citations"][0]["filename"] == "Testing.txt"


# ---------------------------------------------------------------------------
# Test G: shared document library, isolated histories
# ---------------------------------------------------------------------------
def test_g_shared_library_isolated_histories(services, settings, testing_doc):
    """Upload once (chat 1); chats 2 and 3 both retrieve; histories stay
    separate."""
    document, chunk_id = testing_doc
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
        grounded=_grounded_for(document.document_id, "Your name is Alex Example.",
                               "Name: Alex Example"),
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))

    chat1 = _ask(graph, "chat-1", NAME_QUESTION)
    assert chat1["final_response"]["citations"]

    chat2 = _ask(graph, "chat-2", "What is my name?")
    assert chat2["final_response"]["kind"] == "answer"
    assert "Alex Example" in chat2["final_response"]["text"]

    chat3 = _ask(graph, "chat-3", "What is my date of birth?")
    payload3 = chat3["final_response"]
    assert payload3["kind"] == "answer"

    # Histories are isolated: chat-2's transcript contains ONLY its own turn.
    state2 = graph.get_state({"configurable": {"thread_id": "chat-2"}})
    texts2 = [str(m.content) for m in state2.values.get("messages", [])]
    assert any("What is my name?" in t for t in texts2)
    assert not any("date of birth" in t.lower() for t in texts2)


# ---------------------------------------------------------------------------
# Phase-1 repros, now fixed (regression pinning)
# ---------------------------------------------------------------------------
def test_repro_r2_mangled_citation_is_corrected_not_discarded(
    services, settings, testing_doc
):
    """The Phase-1 R2 root cause, now fixed: a free-model answer whose
    citation echoes the FILENAME instead of the chunk id is corrected to
    the real chunk - the answer is kept, never turned into an abstention."""
    document, chunk_id = testing_doc
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95),
        grounded=GroundedAnswer(
            answer="Your name is Alex Example.",
            citations=[Citation(
                document_id="Testing.txt",     # model echoed the filename
                filename="Testing.txt",
                chunk_id="Testing.txt:0",      # mangled id
                supporting_passage="Name: Alex Example",
            )],
            confidence=0.9,
        ),
    )
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-r2", NAME_QUESTION)
    payload = result["final_response"]
    assert payload["insufficient_evidence"] is False
    assert payload["text"] == "Your name is Alex Example."
    citations = payload["citations"]
    assert citations and citations[0]["chunk_id"] == chunk_id
    assert citations[0]["document_id"] == document.document_id
    assert payload.get("citation_recovery")  # the app recorded the fix


def test_repro_r3_unjustified_abstention_gets_one_corrective_retry(
    services, settings, testing_doc
):
    """The Phase-1 R3 root cause, now fixed: a cautious first abstention is
    retried once with explicit feedback; the retry's grounded answer wins."""
    document, chunk_id = testing_doc

    class _RetryAnswersGateway(FakeGateway):
        def __init__(self):
            super().__init__(
                intent=IntentResult(
                    intent=Intent.DOCUMENT_QUESTION, confidence=0.95
                ),
                grounded=GroundedAnswer(
                    answer="I could not find sufficient information about this "
                           "in the documents currently indexed.",
                    citations=[], confidence=0.4, insufficient_evidence=True,
                ),
            )

        def structured(self, schema, messages, run_name=None, metadata=None):
            result = super().structured(schema, messages, run_name, metadata)
            # After the first abstention, the corrective retry answers
            # properly with an ordinal citation.
            if schema.__name__ == "GroundedAnswer" and run_name and \
                    "correction" in run_name:
                self.grounded = GroundedAnswer(
                    answer="Your name is Alex Example.",
                    citations=[Citation(
                        document_id=document.document_id,
                        filename="Testing.txt",
                        chunk_id="S1",
                        supporting_passage="Name: Alex Example",
                    )],
                    confidence=0.9,
                )
                return super().structured(schema, messages, run_name, metadata)
            return result

    services.gateway = _RetryAnswersGateway()
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-r3", NAME_QUESTION)
    payload = result["final_response"]
    assert payload["insufficient_evidence"] is False
    assert payload["text"] == "Your name is Alex Example."
    assert payload["citations"][0]["chunk_id"] == chunk_id


def test_repro_r5_orphan_points_are_reconciled(testing_doc, vector_store,
                                               doc_repo, lexical, settings, db):
    """The Phase-1 R5 root cause, now fixed: a DB<->Qdrant desync leaves an
    orphan point; the startup reconciliation removes it and the UI chunk
    count matches the actual points again."""
    document, _ = testing_doc
    # Simulate the desync: drop the DB row, keep the Qdrant point.
    with db.lock:
        db.connection.execute(
            "DELETE FROM documents WHERE document_id = ?", (document.document_id,)
        )
        db.connection.commit()
    assert vector_store.stats()["points"] == 1  # the orphan remains

    # Startup reconciliation: only indexed documents are valid.
    valid = {
        record.document_id
        for record in doc_repo.list_documents()
        if record.status == "indexed"
    }
    report = vector_store.reconcile(valid)
    lexical.reconcile()
    assert report["orphans_removed"] == 1
    assert vector_store.stats()["points"] == 0
    assert lexical.count_rows() == 0


def test_lexical_fallback_finds_exact_facts_without_vector_match(
    testing_doc, vector_store, lexical, settings, monkeypatch
):
    """The Phase-1 R6 root cause, now fixed: when the vector path returns
    nothing (semantic score below threshold), the FTS5 lexical path still
    finds the exact-fact chunk."""
    document, chunk_id = testing_doc
    from rag.retrieval import Retriever

    retriever = Retriever(
        vector_store, top_k=5, min_score=0.30, lexical_index=lexical
    )
    # Neutralize the vector path entirely: only lexical retrieval remains.
    monkeypatch.setattr(
        vector_store, "search", lambda *a, **k: [], raising=False
    )
    hits = retriever.search("What is my name and date of birth?")
    assert hits, "lexical fallback did not find the chunk"
    assert hits[0].chunk_id == chunk_id
    assert "Name: Alex Example" in hits[0].text


# ---------------------------------------------------------------------------
# Test H: UI canaries (sidebar / composer / processing state)
# ---------------------------------------------------------------------------
def test_h_composer_layout_contract():
    """The composer (follow-up task sections 4-5, 7): attachment previews
    ABOVE the input, ONE plus control (a direct file browser, no popover
    and no caret) on the left, Send on the right, and Stop replacing Send
    while generating."""
    import inspect

    import ui.chat as chat_module

    source = inspect.getsource(chat_module._render_composer)
    # Previews render before the +/input/Send row.
    assert source.index("_render_attachment_chips") < source.index(
        "st.file_uploader"
    )
    # The plus control is a collapsed-label file uploader that opens the
    # file browser directly - the popover (and its caret) is gone.
    assert "st.popover" not in source
    assert source.index("st.file_uploader") < source.index("st.form")
    # Generating state: disabled input with the generating placeholder and
    # a Stop button in the Send position.
    assert source.index("Generating response...") < source.index(
        'key="pkta_composer_stop"'
    )
    # Idle state: the input precedes the Send submit button, and the whole
    # composer honours the processing state.
    assert source.index("st.text_input") < source.index(
        "st.form_submit_button"
    )
    assert "disabled=disabled" in source
    # Stop actually signals the cooperative cancellation event.
    assert "generation.request_stop" in source


def test_h_processing_state_contract():
    """Processing-state design (follow-up task section 7): generation runs
    in a background worker, the monitor fragment consumes it exactly once,
    dedup guards submissions, and the composer is restored on completion
    or cancellation."""
    import inspect

    import ui.chat as chat_module
    import ui.generation as generation_module

    submit = inspect.getsource(chat_module._submit_message)
    monitor = inspect.getsource(chat_module._generation_monitor)
    render = inspect.getsource(chat_module.render)
    # The submitted message starts a background job (unique token) after
    # the dedup fingerprint check.
    assert "pkta_last_request" in submit          # dedup fingerprint
    assert "generation.start_generation" in submit
    # The thinking indicator is visible while the request executes, with
    # the REAL application phase as the label.
    assert "_generation_monitor" in render
    assert "@st.fragment" in monitor
    assert "current_phase" in monitor
    # Terminal jobs are consumed exactly once and trigger a full rerun.
    assert "_consume_terminal_job" in monitor
    assert 'st.rerun(scope="app")' in monitor
    # Stop signals the cancellation event; the worker honours it.
    worker = inspect.getsource(generation_module._run_job)
    assert "begin_generation" in worker
    assert "end_generation" in worker


def test_h_sidebar_hides_native_controls_and_shows_rail():
    """While collapsed, the sidebar is hidden and the left-edge reopen rail
    is rendered; while open, the sidebar toggle sits at its top-left."""
    import inspect

    import app as app_module

    main_source = inspect.getsource(app_module.main)
    assert "if not sidebar_open:" in main_source
    assert "_render_sidebar_rail()" in main_source
    assert 'display:none !important' in app_module.SIDEBAR_COLLAPSED_CSS
    assert 'width:300px' in app_module.SIDEBAR_OPEN_CSS
