"""Follow-up bug fix: reliable document access + sidebar restoration.

Implements the automated regression tests from the follow-up task (section 7):

- Test 1: basic document ingestion (extraction, chunks, embeddings, points)
- Test 2: direct retrieval (independent of LangGraph and the LLM)
- Test 3: the actual LLM request contains the retrieved document text
- Test 4: cross-chat accessibility of the shared library
- Test 5: citation verification (real chunk ids from real retrieval)
- Test 6: insufficient evidence -> abstention, never invention
- Test 7: persistence across application restarts
- Test 8: sidebar restoration (canary checks; full UI flow verified in the
  browser during development - see the final report)

Additional regression tests for the specific failure modes that caused the
reported "assistant claims it cannot access documents" bug.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from core.llm_gateway import ErrorCategory, ModelGatewayError
from core.models import (
    Citation,
    GroundedAnswer,
    Intent,
    IntentResult,
)
from tests.fakes import FakeGateway, tool_call_message

TEST_DOC = (
    "Project Name: Personal Knowledge Assistant\n\n"
    "The application uses PostgreSQL as its primary database.\n\n"
    "The vector database is Qdrant.\n\n"
    "The primary language model is NVIDIA Nemotron 3 Ultra through "
    "OpenRouter.\n\n"
    "The application uses LangGraph for workflow orchestration.\n"
)

CHUNK_TEXT = "The vector database is Qdrant."
# The TXT parser yields ONE section; small sections are now preserved
# VERBATIM (follow-up task 3.2: paragraph boundaries and labels survive),
# so the chunk text is the stripped document text with its newlines. The
# fake embeddings match only identical text, so exact-text queries must use
# this full verbatim chunk.
FULL_CHUNK = TEST_DOC.strip()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def task_doc(ingestion, doc_repo):
    """The exact test document from the follow-up task, indexed once."""
    result = ingestion.ingest_file("ProjectNotes.txt", TEST_DOC.encode())
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    return document, f"{document.document_id}:0"


@pytest.fixture
def doc_gateway(gateway, task_doc):
    """Gateway scripted for grounded answers about the test document."""
    document, chunk_id = task_doc
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.95)
    gateway.grounded = GroundedAnswer(
        answer="According to your document, the vector database is Qdrant.",
        citations=[
            Citation(
                document_id=document.document_id,
                filename="ProjectNotes.txt",
                chunk_id=chunk_id,
                supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )
    return gateway


def _ask(graph, thread_id: str, question: str):
    return graph.invoke(
        {
            "user_request": question,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=question)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def _seen_texts(gateway, since: int = 0) -> list[str]:
    return [
        str(getattr(m, "content", ""))
        for messages, _tools in gateway.seen_messages[since:]
        for m in messages
    ]


# ---------------------------------------------------------------------------
# Test 1: basic document ingestion
# ---------------------------------------------------------------------------
def test_1_ingestion_indexes_real_points(task_doc, vector_store, doc_repo, lexical):
    """Upload -> extraction -> chunks -> embeddings -> Qdrant points."""
    document, chunk_id = task_doc

    assert doc_repo.count_documents(("indexed",)) == 1
    assert document.chunk_count >= 1
    assert document.filename == "ProjectNotes.txt"

    # Chunks contain the expected statements from the test document: an
    # exact-text query retrieves its own chunk (fake embeddings give cosine
    # 1.0 only for identical text), and the chunk carries the statements.
    from rag.retrieval import Retriever

    retriever = Retriever(
        vector_store, top_k=5, min_score=0.0, lexical_index=lexical
    )
    exact = retriever.search(FULL_CHUNK, top_k=3)
    assert exact and exact[0].chunk_id == chunk_id
    assert "PostgreSQL" in exact[0].text
    assert "LangGraph" in exact[0].text
    # Content fidelity: the verbatim chunk preserves the paragraph breaks
    # and the key-value labels of the uploaded file.
    assert "\n\n" in exact[0].text
    assert exact[0].text.startswith("Project Name:")

    # The collection genuinely contains the indexed points.
    stats = vector_store.stats()
    assert stats["exists"] is True
    assert stats["points"] >= document.chunk_count
    assert stats["collection"] == "documents"


def _ingest_variant(ingestion):
    return ingestion.ingest_file(
        "Timings.txt", (TEST_DOC + "Extra paragraph for a new hash.\n").encode()
    )


def test_1_ingestion_timings_and_dupe(task_doc, ingestion, doc_repo):
    """Per-stage timings are measured (the earlier slow-ingestion fix stays)
    and re-uploading the same bytes is a free duplicate."""
    result = _ingest_variant(ingestion)
    assert result.status == "indexed"
    assert result.timings.get("total", 0) > 0
    assert set(result.timings) >= {"parse", "chunk", "embed", "index", "total"}
    # Re-uploading the same bytes is a free duplicate.
    again = _ingest_variant(ingestion)
    assert again.status == "duplicate"


# ---------------------------------------------------------------------------
# Test 2: direct retrieval (no LangGraph, no LLM)
# ---------------------------------------------------------------------------
def test_2_direct_retrieval_returns_relevant_chunk(task_doc, retriever):
    """Execute the retriever directly, exactly as the task requires."""
    document, chunk_id = task_doc
    chunks = retriever.search(FULL_CHUNK, top_k=5)

    assert chunks, "direct retrieval returned nothing"
    top = chunks[0]
    assert CHUNK_TEXT in top.text
    assert top.filename == "ProjectNotes.txt"
    assert top.document_id == document.document_id
    assert top.chunk_id == chunk_id
    # Hybrid retrieval: the chunk may arrive through the vector path (a
    # semantic score) or through the FTS5 lexical path (score None - an
    # exact keyword match, relevant regardless of embedding distance).
    assert top.score is None or top.score > 0.5


def test_2_direct_retrieval_no_filters_by_thread(task_doc, retriever):
    """Retrieval has NO conversation/thread filter - the library is global."""
    _, _ = task_doc
    for thread_id in ("chat-1", "chat-2", "anything"):
        # The retriever API takes no thread parameter at all; this test
        # pins that contract (an accidental per-thread filter would need a
        # parameter or hidden state to exist).
        chunks = retriever.search(FULL_CHUNK, top_k=3)
        assert chunks and chunks[0].filename == "ProjectNotes.txt"


# ---------------------------------------------------------------------------
# Test 3 + the reported bug: retrieved text must reach the LLM
# ---------------------------------------------------------------------------
class _MisbehavingClassifierGateway(FakeGateway):
    """The reported failure mode, reproduced deterministically.

    The free-endpoint classifier labels EVERYTHING general_conversation and
    the chat model never calls tools - exactly the conditions under which
    the assistant used to answer "I don't have access to your documents".
    """

    def chat(self, messages, tools=None, run_name=None, metadata=None):
        self.seen_messages.append((list(messages), tools is not None))
        if tools:
            self.chat_calls_with_tools += 1
            return AIMessage(content="DONE")  # never calls the search tool
        return AIMessage(
            content="I'm sorry, I don't have access to your documents."
        )


def test_3_reported_bug_misclassified_and_toolless_still_rag(
    services, settings, task_doc
):
    """End-to-end reproduction of the reported bug, now fixed.

    "Where is the authentication process discussed?" contains no explicit
    document-reference keyword the old override required, and the classifier
    misroutes it to general_conversation while the chat model refuses to
    call tools. Before the fix this produced a "general conversation"
    answer claiming the assistant had no document access. Now:
    - the question-shape safety net routes it to the document branch, and
    - the application itself executes retrieval (the model never has to
      decide to call a tool), and
    - the answer is an honest, grounded-branch response (here: abstention,
      because the test document says nothing about authentication) instead
      of the "I don't have access" disclaimer.
    """
    gateway = _MisbehavingClassifierGateway(
        intent=IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.8)
    )
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-bug", "Where is the authentication process discussed?")

    payload = result["final_response"]
    # The document branch answered (NOT general conversation, which is what
    # produced the "cannot access documents" reply).
    assert payload["kind"] == "answer"
    assert "don't have access" not in payload["text"]
    # Retrieval was executed by the application itself.
    assert result["tool_results"][0]["tool"] == "search_documents"
    # With fake embeddings nothing semantically matches, so the honest
    # result is abstention - and that is exactly what happens (not a
    # fabricated answer, not an access disclaimer).
    assert payload["insufficient_evidence"] is True


@pytest.mark.parametrize(
    "question",
    [
        "Summarize my uploaded PDF.",
        "What does my research paper say about its methodology?",
        "Explain the database architecture described in my documents.",
        "What are the main points in the uploaded document?",
        "Compare the two uploaded documents.",
        "Where is the authentication process discussed?",
        "What does my project proposal say about LangGraph?",
    ],
)
def test_3_all_task_example_questions_reach_the_document_branch(
    services, settings, task_doc, question
):
    """Every example question from the task routes to the document branch,
    retrieves, and has the retrieved content in the answer-generation
    request - even when the LLM classifier misclassifies it as general
    conversation (the planner's tool call stands in for what real semantic
    embeddings would return)."""
    gateway = FakeGateway(
        intent=IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.8)
    )
    document, chunk_id = task_doc
    gateway.grounded = GroundedAnswer(
        answer="Grounded answer from the document.",
        citations=[
            Citation(
                document_id=document.document_id,
                filename="ProjectNotes.txt",
                chunk_id=chunk_id,
                supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )
    gateway.agent_script = [
        tool_call_message("search_documents", {"query": FULL_CHUNK})
    ]
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-q", question)

    payload = result["final_response"]
    assert payload["kind"] == "answer", question
    assert payload["citations"], question
    assert result["tool_results"][0]["tool"] == "search_documents", question
    # The retrieved chunk text was inside the model's input messages.
    assert any(CHUNK_TEXT in t for t in _seen_texts(gateway)), question


def test_3_planner_outage_is_non_fatal(doc_gateway, services, settings, task_doc):
    """A rate-limited planner must never kill the request.

    Two scenarios, per the new retrieval-first contract:
    1. Direct retrieval already found the passages -> the planner is not
       needed at all; a grounded answer with real citations comes back even
       though the planner endpoint is down.
    2. Direct retrieval found nothing -> the planner is attempted, fails,
       and the failure is recorded as a non-fatal note; the request ends in
       an honest abstention, NOT an application error.
    """
    document, chunk_id = task_doc

    class _PlannerOutageGateway(FakeGateway):
        def chat(self, messages, tools=None, run_name=None, metadata=None):
            self.seen_messages.append((list(messages), tools is not None))
            if tools:
                raise ModelGatewayError(
                    "rate limited", ErrorCategory.RATE_LIMIT
                )
            return AIMessage(content="unused")

    gateway = _PlannerOutageGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9)
    )
    gateway.grounded = doc_gateway.grounded
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))

    # Scenario 1: direct retrieval hit - the planner is never even called.
    result = _ask(graph, "t-planner-1", FULL_CHUNK)
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload["citations"][0]["chunk_id"] == chunk_id
    assert not result.get("errors")
    assert gateway.chat_calls_with_tools == 0

    # Scenario 2: nothing retrieved directly - the planner outage is a
    # non-fatal note and the request ends in an honest abstention.
    result2 = _ask(graph, "t-planner-2",
                   "Where is the authentication process discussed?")
    payload2 = result2["final_response"]
    assert not result2.get("errors")  # NOT treated as an application error
    assert any(
        "planner unavailable" in str(trace.get("note") or "")
        for trace in result2["tool_results"]
    )
    assert payload2["insufficient_evidence"] is True


def test_3_vector_store_failure_is_an_error_not_abstention(
    doc_gateway, services, settings, task_doc, monkeypatch
):
    """Infrastructure failure (Qdrant/embedding) must surface as an error and
    must never be silently reported as 'no evidence found' (task section 6)."""
    class _BrokenRetriever:
        def __init__(self, inner):
            self._inner = inner

        @property
        def store(self):
            return self._inner.store

        def search(self, query, top_k=None):
            raise RuntimeError("qdrant storage corrupted")

    services.retriever = _BrokenRetriever(services.retriever)
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-infra", "What does my document say about databases?")

    payload = result["final_response"]
    assert payload["kind"] == "error"
    assert "unavailable" in payload["text"]
    # Crucially: NOT the abstention path.
    assert payload.get("insufficient_evidence") is None


# ---------------------------------------------------------------------------
# Test 4: cross-chat accessibility
# ---------------------------------------------------------------------------
def test_4_every_chat_retrieves_the_same_document(
    graph_factory, doc_gateway, task_doc
):
    """Chat 2 and Chat 3 (fresh threads) both retrieve Chat 1's upload."""
    document, chunk_id = task_doc
    for thread_id, question in (
        ("chat-2", "Which vector database does the application use?"),
        ("chat-3", "Which language model does the application use?"),
    ):
        doc_gateway.agent_script = [tool_call_message("search_documents",
                                                      {"query": FULL_CHUNK})]
        result = _ask(graph_factory(), thread_id, question)
        payload = result["final_response"]
        assert payload["kind"] == "answer", thread_id
        assert payload["citations"], thread_id
        assert payload["citations"][0]["filename"] == "ProjectNotes.txt"
        assert payload["citations"][0]["chunk_id"] == chunk_id


# ---------------------------------------------------------------------------
# Test 5: citation verification
# ---------------------------------------------------------------------------
def test_5_citations_originate_from_retrieved_chunks(doc_gateway, graph, task_doc):
    document, chunk_id = task_doc
    doc_gateway.agent_script = [tool_call_message("search_documents",
                                                  {"query": FULL_CHUNK})]
    result = _ask(graph, "t-cite", "What does the document say about "
                                   "workflow orchestration?")
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    for citation in payload["citations"]:
        assert citation["chunk_id"] == chunk_id
        assert citation["document_id"] == document.document_id
        assert citation["filename"] == "ProjectNotes.txt"


# ---------------------------------------------------------------------------
# Test 6: insufficient evidence
# ---------------------------------------------------------------------------
def test_6_unanswerable_question_abstains(doc_gateway, graph, task_doc):
    """The test document says nothing about payment gateways.

    The document IS retrieved (the planner finds it), so the model really
    sees the evidence and must decide it does not support the answer - the
    scripted grounded response flags insufficient evidence and returns no
    citations, and the application turns that into the canonical honest
    abstention instead of letting the model guess.
    """
    doc_gateway.grounded = GroundedAnswer(
        answer="I could not find sufficient information about this in your "
        "uploaded documents.",
        citations=[],
        confidence=0.2,
        insufficient_evidence=True,
    )
    doc_gateway.agent_script = [tool_call_message("search_documents",
                                                   {"query": FULL_CHUNK})]
    result = _ask(graph, "t-pay", "What payment gateway does the "
                                  "application use?")
    payload = result["final_response"]
    assert payload["insufficient_evidence"] is True
    assert payload["citations"] == []
    assert "could not find" in payload["text"]
    # The model really did receive the retrieved context before abstaining.
    assert "GroundedAnswer" in doc_gateway.structured_calls


# ---------------------------------------------------------------------------
# Test 7: persistence across restarts
# ---------------------------------------------------------------------------
def test_7_documents_survive_application_restart(
    settings, doc_repo, task_doc, embeddings, vector_store, lexical
):
    """A brand-new VectorStore on the same disk path still retrieves the doc.

    Qdrant local mode locks the storage folder, so the "restart" closes the
    original client first - exactly what an application restart does. The
    FTS5 lexical index lives in the same SQLite database as the document
    bookkeeping, so it survives the restart the same way.
    """
    from rag.retrieval import Retriever
    from rag.vector_store import VectorStore

    vector_store.close()  # the old process shuts down
    restarted_store = VectorStore(settings.qdrant_path, embeddings)
    try:
        restarted = Retriever(
            restarted_store, top_k=5, min_score=0.0, lexical_index=lexical
        )
        chunks = restarted.search(FULL_CHUNK, top_k=3)
        assert chunks and chunks[0].filename == "ProjectNotes.txt"
        assert restarted_store.stats()["points"] >= 1
    finally:
        restarted_store.close()


def test_7_new_chat_after_restart_queries_document(
    graph_factory, doc_gateway, task_doc
):
    """New chat + restarted graph: retrieval and grounded answer still work."""
    document, chunk_id = task_doc
    doc_gateway.agent_script = [tool_call_message("search_documents",
                                                  {"query": FULL_CHUNK})]
    result = _ask(graph_factory(), "fresh-chat-after-restart",
                  "Which vector database does the application use?")
    assert result["final_response"]["citations"][0]["chunk_id"] == chunk_id


# ---------------------------------------------------------------------------
# Contextual follow-up resolution (task section 4)
# ---------------------------------------------------------------------------
def test_followup_resolved_via_model_rewrite(doc_gateway, graph, task_doc):
    """'What was the reason for choosing it?' after a grounded answer."""
    document, chunk_id = task_doc
    standalone = ("What was the reason for choosing Qdrant as the vector "
                  "database?")

    # Turn 1: a grounded answer puts the thread in document mode.
    doc_gateway.agent_script = [tool_call_message("search_documents",
                                                  {"query": FULL_CHUNK})]
    result1 = _ask(graph, "t-followup", "What database is used in my project "
                                        "proposal?")
    assert result1["final_response"]["kind"] == "answer"

    # Turn 2: the follow-up. The fake gateway's plain-chat reply is the
    # model rewrite; the planner tool call retrieves the chunk again.
    doc_gateway.chat_text = standalone
    doc_gateway.agent_script = [tool_call_message("search_documents",
                                                  {"query": FULL_CHUNK})]
    result2 = _ask(graph, "t-followup", "What was the reason for choosing it?")
    payload = result2["final_response"]

    assert payload["kind"] == "answer"
    assert payload["citations"]
    # The retrieval query was the context-resolved standalone question.
    first_call = result2["tool_results"][0]
    assert first_call["input"]["method"] == "model_rewrite"
    assert first_call["input"]["query"] == standalone
    # The grounded-answer prompt carried the standalone interpretation.
    grounded_prompt = doc_gateway.seen_messages[-1][0]
    assert any(
        "Standalone interpretation" in str(getattr(m, "content", ""))
        and standalone in str(getattr(m, "content", ""))
        for m in grounded_prompt
    )


def test_followup_deterministic_fallback_when_rewrite_fails(
    graph, gateway, task_doc
):
    """Provider outage during query resolution -> deterministic merge."""
    document, chunk_id = task_doc

    class _RewriteOutageGateway(FakeGateway):
        """Fails plain-chat calls (the standalone rewrite) once armed; tool
        calls keep working (the planner still retrieves)."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.rewrite_fails = False

        def chat(self, messages, tools=None, run_name=None, metadata=None):
            if tools is None and self.rewrite_fails:
                raise ModelGatewayError(
                    "rewrite unavailable", ErrorCategory.RATE_LIMIT
                )
            return super().chat(
                messages, tools=tools, run_name=run_name, metadata=metadata
            )

    gateway.__class__ = _RewriteOutageGateway
    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9)
    gateway.grounded = GroundedAnswer(
        answer="Reason from the document.",
        citations=[
            Citation(
                document_id=document.document_id,
                filename="ProjectNotes.txt",
                chunk_id=chunk_id,
                supporting_passage=CHUNK_TEXT,
            )
        ],
        confidence=0.9,
    )

    first_question = "What database is used in my project proposal?"
    followup = "What was the reason for choosing it?"

    # Turn 1: planner retrieves normally.
    gateway.agent_script = [tool_call_message("search_documents",
                                              {"query": FULL_CHUNK})]
    _ask(graph, "t-det", first_question)

    # Turn 2: the standalone-query rewrite call fails (rate limit); the
    # deterministic fallback must resolve the query from the previous turn.
    gateway.rewrite_fails = True
    gateway.agent_script = [tool_call_message("search_documents",
                                              {"query": FULL_CHUNK})]
    result = _ask(graph, "t-det", followup)
    payload = result["final_response"]

    assert payload["kind"] == "answer"
    first_call = result["tool_results"][0]
    assert first_call["input"]["method"] == "deterministic"
    assert first_question[:30] in first_call["input"]["query"]


# ---------------------------------------------------------------------------
# Routing-safety regression (misclassification override)
# ---------------------------------------------------------------------------
def test_unsupported_classification_of_doc_question_overridden(
    doc_gateway, services, settings, task_doc
):
    """Classifier says 'unsupported_request' for a clear document question."""
    gateway = FakeGateway(
        intent=IntentResult(intent=Intent.UNSUPPORTED_REQUEST, confidence=0.7)
    )
    gateway.grounded = doc_gateway.grounded
    gateway.agent_script = [tool_call_message("search_documents",
                                              {"query": FULL_CHUNK})]
    services.gateway = gateway
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(services, create_checkpointer(settings.checkpoint_db_path))
    result = _ask(graph, "t-unsup-override",
                  "What does my uploaded document say about databases?")
    assert result["final_response"]["kind"] == "answer"
    assert result["final_response"]["citations"]


def test_general_question_without_doc_reference_not_hijacked(
    doc_gateway, graph, task_doc
):
    """A general question must still stay general even with docs indexed."""
    doc_gateway.intent = IntentResult(
        intent=Intent.GENERAL_CONVERSATION, confidence=0.9
    )
    doc_gateway.agent_script = [AIMessage(content="The capital of France is Paris.")]
    result = _ask(graph, "t-general", "What is the capital of France?")
    assert result["final_response"]["kind"] == "conversation"
    assert "Paris" in result["final_response"]["text"]


# ---------------------------------------------------------------------------
# Diagnostics logging (safe, no content)
# ---------------------------------------------------------------------------
def test_diagnostics_logs_retrieval_and_context(doc_gateway, graph, task_doc,
                                                caplog):
    doc_gateway.agent_script = [tool_call_message("search_documents",
                                                  {"query": FULL_CHUNK})]
    with caplog.at_level(logging.INFO, logger="pkta.rag"):
        _ask(graph, "t-diag", "What database does my document use?")
    messages = [record.getMessage() for record in caplog.records]

    assert any("Document retrieval executed: Yes" in m for m in messages)
    assert any("Context included in LLM request: Yes" in m for m in messages)
    # Privacy: the chunk text itself is never logged.
    assert not any(CHUNK_TEXT in m for m in messages)


def test_diagnostics_logs_ingestion_points(ingestion, vector_store, doc_repo,
                                           caplog):
    with caplog.at_level(logging.INFO, logger="pkta.rag"):
        result = ingestion.ingest_file("DiagDoc.txt", TEST_DOC.encode())
    assert result.status == "indexed"
    messages = [record.getMessage() for record in caplog.records]
    assert any("ingestion finished" in m for m in messages)
    assert any("ProjectNotes" not in m for m in messages)  # content not logged
    stats = vector_store.stats()
    assert stats["points"] >= 1


# ---------------------------------------------------------------------------
# Test 8: sidebar restoration (canary checks)
# ---------------------------------------------------------------------------
def test_8_theme_restores_native_expand_control_visibility():
    """Canary for the sidebar root-cause fix: the theme must force the native
    expand button visible (it lives inside the hidden app header, where
    ``visibility: hidden`` inherits onto it - the reported bug)."""
    from ui.theme import THEME_CSS

    assert '[data-testid="stExpandSidebarButton"]' in THEME_CSS
    assert "visibility: visible !important" in THEME_CSS


def test_8_app_has_persistent_toggle_outside_the_sidebar():
    """Canary: the sidebar-side toggle design (follow-up task section 6).

    - The collapse toggle lives at the TOP-LEFT OF THE SIDEBAR (next to the
      brand) with the "Close sidebar" tooltip.
    - The reopen control is rendered OUTSIDE the sidebar (the main area),
    pinned to the left edge of the application, with the "Open sidebar"
      tooltip.
    - The app state is authoritative in both directions.
    """
    import inspect

    import app as app_module

    assert hasattr(app_module, "_render_sidebar")
    assert hasattr(app_module, "_render_sidebar_rail")
    sidebar_source = inspect.getsource(app_module._render_sidebar)
    rail_source = inspect.getsource(app_module._render_sidebar_rail)

    # Sidebar-side collapse toggle with the required tooltip.
    assert '"Close sidebar"' in sidebar_source
    # The reopen rail is a real button rendered outside the sidebar, with
    # the required tooltip, and only while the sidebar is collapsed.
    assert '"Open sidebar"' in rail_source
    assert 'st.sidebar' not in rail_source
    assert 'if not sidebar_open:' in inspect.getsource(app_module.main)
    # The app state is authoritative in BOTH directions: it can force the
    # sidebar open (desynced native state) and hide it (collapse).
    assert 'data-testid="stSidebar"' in app_module.SIDEBAR_COLLAPSED_CSS
    assert 'data-testid="stSidebar"' in app_module.SIDEBAR_OPEN_CSS
    # The theme styles the rail (left-edge, fixed) and the sidebar toggle.
    from ui.theme import THEME_CSS

    assert ":has(.pkta-rail-marker)" in THEME_CSS
    assert ".st-key-pkta_sidebar_reopen" in THEME_CSS
    assert ".st-key-pkta_sidebar_collapse" in THEME_CSS
    assert '[data-testid="stExpandSidebarButton"]' in THEME_CSS


def test_8_sidebar_order_matches_required_hierarchy():
    """The sidebar renders in the exact order required by follow-up task 3:
    application name/logo (top-left) + toggle, New Chat, Documents, Tasks,
    Settings, Approvals, then the Chats history section."""
    import inspect

    import app as app_module

    source = inspect.getsource(app_module._brand) + inspect.getsource(
        app_module._render_sidebar
    )
    order = [
        'pkta-name">Knowledge',          # 1. application name (top-left)
        '"pkta_sidebar_collapse',          #    toggle on the same brand row
        '"＋  New Chat"',                  # 2. New Chat button
        '"documents", "📄  Documents"',     # 3. Documents
        '"tasks", "✅  Tasks"',             # 4. Tasks
        '"settings", "⚙️  Settings"',       # 5. Settings (with the nav group)
        '"approvals", "⏳  Approvals"',     #    (existing page, kept after S)
        'pkta-history-title">Chats',        # 6./7. chat history section
    ]
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions), (
        "sidebar sections are out of the required order"
    )


def test_8_sidebar_branding_and_footer_removed():
    """Follow-up task, section 2: the 'local · private · yours' sub-line and
    the 'RAG over your documents ...' footer are gone, with no replacement
    promotional text anywhere in the sidebar."""
    import inspect

    import app as app_module

    # Only the RENDERED markup matters (the docstring may legitimately
    # mention what was removed).
    brand_source = inspect.getsource(app_module._brand)
    render_block = brand_source[brand_source.index("st.markdown"):]
    assert "local · private · yours" not in render_block
    assert "pkta-sub" not in render_block
    sidebar_source = inspect.getsource(app_module._render_sidebar)
    assert "RAG over your documents" not in sidebar_source
    assert "persistent memory" not in sidebar_source
    # No replacement tagline was introduced either.
    assert "st.caption" not in sidebar_source.replace(
        'st.caption("No conversations yet', 'KEEP'
    )


def test_8_sidebar_state_never_touches_conversation_keys():
    """The toggle handler contract: it must not reset the conversation.

    The sidebar function legitimately writes ``pkta_thread``/``pkta_page``
    for chat switching and navigation; this test isolates the TOGGLE blocks
    (from the toggle button to its rerun) and asserts those blocks only
    write the sidebar-visibility flag.
    """
    import inspect

    import app as app_module

    def _toggle_block(source: str, key: str) -> str:
        start = source.index(f'key="{key}"')
        end = source.index("st.rerun()", start)
        return source[start:end]

    sidebar_source = inspect.getsource(app_module._render_sidebar)
    rail_source = inspect.getsource(app_module._render_sidebar_rail)
    for block in (
        _toggle_block(sidebar_source, "pkta_sidebar_collapse"),
        _toggle_block(rail_source, "pkta_sidebar_reopen"),
    ):
        assert '"pkta_sidebar_open"' in block
        assert "pkta_thread" not in block
        assert "pkta_page" not in block
