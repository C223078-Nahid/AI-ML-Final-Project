"""Workflow nodes (concepts C7, E2).

Each node is a small function (state -> partial state update) closing over a
``GraphServices`` bundle.  Nodes never raise for model-level failures: they
record errors in state so the workflow can route to final_response with a
clear message.

The human_approval node calls LangGraph ``interrupt()``.  Per the LangGraph
interrupt rules: it is NOT wrapped in try/except, it performs no side
effects before pausing, and its payload is JSON-serializable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt
from pydantic import ValidationError

from config.settings import Settings
from core import cancellation, datetime_util, guardrails, prompts
from core.llm_gateway import ModelGateway, ModelGatewayError
from core.models import (
    ClaimAssessmentBatch,
    ExtractedClaims,
    FactCheckReport,
    GroundedAnswer,
    Intent,
    IntentResult,
    RetrievedChunk,
    TaskProposal,
    TaskStatus,
    UserPreference,
)
from database.conversations import ConversationRepository
from database.documents import DocumentRepository
from database.memory import MemoryRepository
from database.tasks import TaskRepository
from rag import diagnostics
from rag.retrieval import Retriever
from tools.document_tools import make_search_documents_tool
from tools.task_tools import make_create_task_tool, make_list_tasks_tool


@dataclass
class GraphServices:
    """All collaborators the graph nodes need, injected for testability."""

    gateway: ModelGateway
    retriever: Retriever
    task_repo: TaskRepository
    memory_repo: MemoryRepository
    conversation_repo: ConversationRepository
    settings: Settings
    # The shared document library (global across every chat thread). Used
    # by the deterministic routing safety net in intent_router.
    doc_repo: DocumentRepository | None = None
    # Centralized tool registry (follow-up task 4, section 6.2).  Optional:
    # when absent, ``make_nodes`` builds one lazily from the services above
    # so direct test construction keeps working.
    tool_registry: Any = None
    # Provider-independent web search service (follow-up task 4, section
    # 9).  Shared with the registry tools; None builds one from settings.
    web_search: Any = None


#: User-visible text of the stopped marker message (section 7.2).
STOPPED_TEXT = (
    "Response stopped by user. The request was cancelled; nothing else was "
    "executed afterwards. You can send a new message at any time."
)


def _cancelled() -> dict:
    """Standard node short-circuit update when Stop was pressed."""
    cancellation.report_phase("Stopping…")
    return {"cancelled": True}


def _phase(label: str) -> None:
    """Report an actual application progress phase to the UI."""
    cancellation.report_phase(label)


def _prefs(state: dict) -> list[UserPreference]:
    return [UserPreference(**p) for p in (state.get("preferences") or [])]


def _history(state: dict) -> list[Any]:
    return state.get("messages") or []


def _previous_user_turn(history: list[Any], current_request: str) -> str | None:
    """The user message immediately before the current one, if any."""
    for message in reversed(history or []):
        if message.__class__.__name__ != "HumanMessage":
            continue
        text = str(message.content or "").strip()
        if not text or text == (current_request or "").strip():
            continue
        return text
    return None


def _resolve_search_query(state: dict, gateway: ModelGateway) -> tuple[str, str]:
    """Build a standalone retrieval query for the current request.

    Returns ``(query, method)`` where method is one of:
    - ``direct``:            the request is self-contained, use it verbatim;
    - ``model_rewrite``:     a follow-up, resolved into a standalone question
                             by one bounded model call;
    - ``deterministic``:     a follow-up where the model call failed; the
                             previous user turn is prepended instead.

    This is what makes contextual follow-ups ("What was the reason for
    choosing it?") retrievable: the raw pronoun question alone would
    retrieve nothing meaningful.
    """
    request = state.get("user_request") or ""
    history = _history(state)
    previous = _previous_user_turn(history, request)
    if not (previous and guardrails.looks_like_followup(request)):
        return request, "direct"

    try:
        response = gateway.chat(
            prompts.standalone_query_messages(request, history[-8:]),
            run_name="resolve_search_query",
            metadata={"prompt_version": prompts.PROMPT_VERSIONS["standalone_query"]},
        )
        rewritten = str(response.content or "").strip().strip('"').strip()
        if 8 <= len(rewritten) <= 400:
            return rewritten, "model_rewrite"
    except ModelGatewayError as exc:
        diagnostics.log_retrieval_failure("standalone_query_resolution", exc)

    merged = f"{previous[:200]} {request}"[:400]
    return merged, "deterministic"


def make_nodes(services: GraphServices) -> dict[str, Callable[[dict], dict]]:
    gateway = services.gateway
    retriever = services.retriever
    task_repo = services.task_repo
    settings = services.settings

    # ---- shared collaborators (follow-up task 4, sections 6, 8, 9) --------
    # ONE web-search service per application: the agent tools and the
    # fact-check workflow share its per-run URL allowlist (each generation
    # worker runs in its own thread; reset_run_scope() is called per
    # request).
    search_service = getattr(services, "web_search", None)
    if search_service is None:
        from core.web_search import build_search_service

        search_service = build_search_service(settings)

    def _request_registry() -> tuple:
        """A per-request tool registry over the centralized factory.

        The registry is built from the SAME ``build_default_registry``
        factory everywhere (single source of truth); it is rebuilt per
        request so run-scoped state - the retrieval collector used for
        citation bookkeeping - never leaks across requests or threads.
        Returns ``(registry, bindable_tools)``.
        """
        from tools.registry import build_default_registry

        registry = build_default_registry(
            retriever=retriever,
            doc_repo=services.doc_repo,
            task_repo=task_repo,
            settings=settings,
            search_service=search_service,
            collector={},
        )
        if getattr(settings, "llm_tool_calling_enabled", True):
            return registry, registry.bindable_tools()
        return registry, []

    def _tool_names(bindable: list) -> list[str]:
        return [getattr(t, "name", str(t)) for t in bindable]

    # ------------------------------------------------------------------ 1
    def input_guardrail(state: dict) -> dict:
        """Validate user input and reset per-turn state (C5)."""
        _phase("Thinking…")
        updates: dict[str, Any] = {
            "retrieved_documents": [],
            "retrieval_query": "",
            "task_proposal": None,
            "operation_id": None,
            "approval_status": "not_required",
            "task_id": None,
            "tool_results": [],
            "factcheck_claims": None,
            "final_response": {},
            "errors": [],
            "cancelled": False,
        }
        # Long-term memory is loaded fresh at the start of every turn (E3).
        updates["preferences"] = [
            p.model_dump(mode="json") for p in services.memory_repo.list_preferences()
        ]

        request = state.get("user_request") or ""
        result = guardrails.validate_user_message(request, settings)
        if not result.ok:
            if result.refusal:
                updates["final_response"] = {
                    "kind": "refusal",
                    "text": result.refusal,
                    "matched_rule": result.matched_rule,
                }
            else:
                updates["errors"] = [result.error or "Invalid input."]
        return updates

    # ------------------------------------------------------------------ 2
    def intent_router(state: dict) -> dict:
        """Classify the request with structured output (C1).

        Follow-up task 4: seven routing categories with deterministic
        safety nets, in this precedence:

        1. **Datetime safety net (adapted, section 7)** - a question about
           the current date/time is NEVER answered from documents or model
           training data.  It is routed to the tool-calling agent, where
           the MODEL decides to invoke the registered get_current_datetime
           tool (no more deterministic datetime graph node).  Only task
           operations win over it.
        2. **Fact-check safety net (section 8.3)** - an explicit
           verification request reaches the dedicated fact-check workflow
           even when the classifier mislabels it; fact-checking is never
           triggered for ordinary document questions.
        3. **Document safety net** - document-shaped questions misrouted to
           general_conversation / general_knowledge / unsupported_request
           are pulled back to the document branch when the shared library
           has indexed documents (the original "model cannot access my
           documents" bug).
        4. **Follow-up stickiness** - a contextual follow-up inside a thread
           that is already answering document questions stays on the
           document branch.
        5. If the classification call itself fails, the deterministic
           heuristic classifier (datetime-first) is used instead of failing
           the whole request.

        Task operations are never redirected into the RAG branch.
        """
        if cancellation.cancellation_requested():
            return _cancelled()
        _phase("Understanding your request…")
        request = state.get("user_request") or ""
        library_has_documents = (
            services.doc_repo is not None
            and services.doc_repo.count_documents(("indexed",)) > 0
        )
        heuristic = guardrails.heuristic_intent(request)
        datetime_hint = guardrails.looks_like_datetime_query(request)
        factcheck_hint = guardrails.looks_like_factcheck_request(request)
        followup_in_doc_thread = (
            library_has_documents
            and guardrails.looks_like_followup(request)
            and guardrails.thread_in_document_mode(_history(state))
        )
        doc_question_hint = (
            library_has_documents
            and heuristic == "document_question"
        )

        try:
            intent = gateway.structured(
                IntentResult,
                prompts.intent_messages(request, _history(state)),
                run_name="classify_intent",
                metadata={
                    "prompt_version": prompts.PROMPT_VERSIONS["intent_classification"]
                },
            )
        except ModelGatewayError as exc:
            fallback_intent = heuristic
            if fallback_intent == "fact_check" and not library_has_documents:
                fallback_intent = "document_question"
            if followup_in_doc_thread and fallback_intent == "general_conversation":
                fallback_intent = "document_question"
            diagnostics.log_routing(fallback_intent, "deterministic fallback")
            return {
                "intent": IntentResult(
                    intent=Intent(fallback_intent), confidence=0.3
                ).model_dump(mode="json"),
                "tool_results": [
                    {
                        "tool": "intent_fallback",
                        "input": {"reason": str(exc)[:200]},
                        "result": fallback_intent,
                    }
                ],
            }

        routing_source = "model classifier"
        if datetime_hint and intent.intent not in (
            Intent.LIST_TASKS,
            Intent.CREATE_TASK,
            Intent.FACT_CHECK,
        ):
            # Keep the datetime intent label (it describes the request) but
            # the ROUTE now leads to the tool-calling agent: the model
            # invokes get_current_datetime itself (follow-up task 4,
            # section 7 - no deterministic answer node).
            intent = intent.model_copy(
                update={
                    "intent": Intent.CURRENT_DATETIME,
                    "requires_retrieval": False,
                    "requires_tool": True,
                }
            )
            routing_source = "datetime safety net (tool-calling agent)"
        elif factcheck_hint and library_has_documents and intent.intent in (
            Intent.DOCUMENT_QUESTION,
            Intent.GENERAL_CONVERSATION,
            Intent.GENERAL_KNOWLEDGE,
        ):
            intent = intent.model_copy(
                update={
                    "intent": Intent.FACT_CHECK,
                    "requires_retrieval": True,
                    "requires_tool": True,
                }
            )
            routing_source = "fact-check safety net"
        elif (doc_question_hint or followup_in_doc_thread) and intent.intent in (
            Intent.GENERAL_CONVERSATION,
            Intent.GENERAL_KNOWLEDGE,
            Intent.UNSUPPORTED_REQUEST,
        ):
            intent = intent.model_copy(
                update={
                    "intent": Intent.DOCUMENT_QUESTION,
                    "requires_retrieval": True,
                    "requires_tool": True,
                }
            )
            routing_source = "document safety net"

        updates: dict[str, Any] = {"intent": intent.model_dump(mode="json")}
        if routing_source != "model classifier":
            diagnostics.log_routing(intent.intent.value, routing_source)
        if intent.intent == Intent.UNSUPPORTED_REQUEST:
            updates["final_response"] = {
                "kind": "refusal",
                "text": prompts.refusal_text(
                    "That request is outside the operations this assistant "
                    "supports.",
                    "I can search your indexed documents, fact-check their "
                    "claims, list your tasks, propose new tasks (each "
                    "proposal needs your explicit approval), and search the "
                    "public web when independent verification helps.",
                ),
            }
        return updates

    # ------------------------------------------------------------------ 3
    def document_retrieval(state: dict) -> dict:
        """Deterministic retrieval-first RAG node.

        Follow-up task, section 4: document access must NOT depend on the
        model deciding to invoke a search tool. The order is therefore:

        0. Empty-library check - a distinct, honest message (application
           state, not a missing-evidence abstention and not an error).
        1. Resolve the search query against the conversation context
           (follow-ups like "What was the reason for choosing it?" are
           rewritten into standalone questions; model-assisted with a
           deterministic fallback).
        2. DIRECT retrieval executed by the application itself - this always
           runs, before any planner LLM call.
        3. Optional bounded planner loop (existing behaviour, kept for
           multi-hop questions) - only when direct retrieval was weak, and
           a planner failure is non-fatal: the direct results still reach
           the answer generation.
        4. Rank, apply the context budget, log safe diagnostics, return.

        Infrastructure failures (embedding, Qdrant) are recorded as errors
        so the final response is an error - never silently "no evidence".
        """
        if cancellation.cancellation_requested():
            return _cancelled()
        tool_trace: list[dict] = []

        # -- 0. empty library: an honest, distinct answer, not a fake abstention
        if (
            services.doc_repo is not None
            and services.doc_repo.count_documents(("indexed",)) == 0
        ):
            diagnostics.log_library_status(services.retriever.store, 0)
            return {
                "final_response": {
                    "kind": "answer",
                    "text": guardrails.EMPTY_LIBRARY_TEXT,
                    "citations": [],
                    "insufficient_evidence": True,
                    "library_empty": True,
                },
                "tool_results": [
                    {"tool": "search_documents", "input": {}, "note": "library empty"}
                ],
            }

        # -- 1. context-aware query resolution
        _phase("Searching uploaded documents…")
        query, query_method = _resolve_search_query(state, gateway)

        # -- 2. deterministic direct retrieval (always executed)
        _phase("Retrieving relevant passages…")
        collector: dict[str, dict] = {}
        try:
            direct_chunks = retriever.search(query)
        except Exception as exc:  # noqa: BLE001 - classified below
            diagnostics.log_retrieval_failure("direct_retrieval", exc)
            return {
                "errors": [
                    f"Document search is currently unavailable "
                    f"({type(exc).__name__}): {str(exc)[:200]}"
                ]
            }
        for chunk in direct_chunks:
            collector[chunk.chunk_id] = chunk.model_dump()
        tool_trace.append(
            {
                "tool": "search_documents",
                "input": {"query": query, "method": query_method},
                "new_results": len(direct_chunks),
            }
        )
        diagnostics.log_retrieval(query_method, direct_chunks)

        # -- 3. optional planner supplement (only when direct retrieval found
        #    NOTHING). With hybrid retrieval (vector + FTS5 lexical), a
        #    non-empty direct result already found relevant passages; the
        #    previous "< 2 chunks" rule re-ran the planner for every
        #    single-chunk document, wasted a model call, and was the source
        #    of the reported retrieval-count inconsistencies.
        if not collector:
            search_tool = make_search_documents_tool(
                retriever, settings.retrieval_top_k, collector
            )
            loop_messages = prompts.retrieval_planner_messages(
                query, settings.max_tool_iterations
            )
            total_calls = 0
            try:
                for _ in range(settings.max_tool_iterations):
                    if cancellation.cancellation_requested():
                        # Stop pressed inside the planner loop: no further
                        # model calls, no further tool executions.
                        return {**_cancelled(), "retrieved_documents": [],
                                "retrieval_query": query,
                                "tool_results": tool_trace}
                    if total_calls >= settings.max_total_tool_calls:
                        break
                    response = gateway.chat(
                        loop_messages,
                        tools=[search_tool],
                        run_name="retrieval_planner",
                        metadata={
                            "prompt_version": prompts.PROMPT_VERSIONS[
                                "retrieval_planner"
                            ]
                        },
                    )
                    if not response.tool_calls:
                        break
                    for tool_call in response.tool_calls:
                        if total_calls >= settings.max_total_tool_calls:
                            break
                        if cancellation.cancellation_requested():
                            return {**_cancelled(), "retrieved_documents": [],
                                    "retrieval_query": query,
                                    "tool_results": tool_trace}
                        total_calls += 1
                        before = len(collector)
                        try:
                            output = search_tool.invoke(tool_call["args"])
                        except Exception as exc:  # noqa: BLE001 - fed back to model
                            output = f"Tool error: {exc}"
                        loop_messages.append(
                            ToolMessage(
                                content=str(output), tool_call_id=tool_call["id"]
                            )
                        )
                        tool_trace.append(
                            {
                                "tool": "search_documents",
                                "input": {
                                    "query": str(tool_call["args"].get("query", ""))
                                },
                                "new_results": len(collector) - before,
                            }
                        )
            except ModelGatewayError as exc:
                # Planner failure is NOT fatal: direct retrieval already ran.
                diagnostics.log_retrieval_failure("retrieval_planner", exc)
                tool_trace.append(
                    {
                        "tool": "search_documents",
                        "input": {},
                        "note": "planner unavailable; direct retrieval used",
                        "error": str(exc)[:160],
                    }
                )

        ranked = sorted(
            collector.values(), key=lambda c: -(c.get("score") or 0.0)
        )
        limited = guardrails.limit_context(
            [RetrievedChunk(**c) for c in ranked], settings.max_context_chars
        )
        return {
            "retrieved_documents": [c.model_dump(mode="json") for c in limited],
            "retrieval_query": query,
            "tool_results": tool_trace,
        }

    # ------------------------------------------------------------------ 4
    def grounded_answer(state: dict) -> dict:
        """Generate an evidence-based answer and validate citations (C6).

        Follow-up task, section 4. Three layers make sure a correct answer
        is never thrown away:

        1. Passages are rendered with short ordinal source ids (S1, S2, ...)
           which free endpoints echo reliably; citation validation maps
           them back to the real chunks and *corrects* citations that
           reference the right passage through filename or quoted text.
        2. A flawed first reply (model abstained despite evidence, or
           citations could not be resolved) gets ONE bounded corrective
           retry with explicit feedback - the same pattern the gateway
           already uses for malformed JSON.
        3. Only when the retry also fails does the response become an honest
           abstention (model twice said the passages lack the answer) or an
           answer without citations flagged as unverified - never a silent
           discard of a grounded answer.
        """
        if cancellation.cancellation_requested():
            return _cancelled()
        _phase("Preparing response…")
        chunks = [RetrievedChunk(**d) for d in state.get("retrieved_documents") or []]
        if not chunks:
            return {
                "final_response": {
                    "kind": "answer",
                    "text": guardrails.ABSTENTION_TEXT,
                    "citations": [],
                    "insufficient_evidence": True,
                }
            }
        question = state.get("user_request") or ""
        standalone = state.get("retrieval_query") or ""
        description = gateway.describe() if hasattr(gateway, "describe") else {}
        provider = description.get("primary_provider", "unknown")
        model_name = description.get("primary_model", "unknown")

        try:
            answer = gateway.structured(
                GroundedAnswer,
                prompts.grounded_answer_messages(
                    question, chunks, _prefs(state), standalone_question=standalone
                ),
                run_name="grounded_answer",
                metadata={
                    "prompt_version": prompts.PROMPT_VERSIONS["grounded_answer"]
                },
            )
        except ModelGatewayError as exc:
            diagnostics.log_model_invocation(
                "grounded_answer", provider, model_name, ok=False, error=str(exc)
            )
            return {"errors": [f"Answer generation failed: {exc}"]}

        # The retrieved passages WERE in the final model request (the
        # capture hook and the seen-message assertions in the test suite
        # verify this end-to-end).
        diagnostics.log_context_construction(chunks, provider, model_name)
        diagnostics.log_model_invocation("grounded_answer", provider, model_name, ok=True)

        validated = guardrails.validate_citations(answer, chunks)
        if validated.problem is not None:
            # ONE corrective retry with explicit feedback about the problem.
            problem_text = {
                "model_abstained": (
                    "you marked the answer as insufficient evidence, but "
                    "retrieved passages are available and may contain the "
                    "answer"
                ),
                "no_citations": (
                    "you did not cite any source even though you answered "
                    "from the passages"
                ),
                "unresolved_citations": (
                    "the citations did not reference any of the provided "
                    "sources"
                ),
            }.get(validated.problem, validated.problem)
            try:
                retry = gateway.structured(
                    GroundedAnswer,
                    prompts.grounded_correction_messages(
                        question, chunks, problem_text
                    ),
                    run_name="grounded_answer_correction",
                    metadata={
                        "prompt_version": prompts.PROMPT_VERSIONS[
                            "grounded_correction"
                        ]
                    },
                )
                diagnostics.log_model_invocation(
                    "grounded_answer_correction", provider, model_name, ok=True
                )
                validated = guardrails.validate_citations(retry, chunks)
            except ModelGatewayError as exc:
                # Retry failure is not fatal: the first validated outcome
                # stands (and its problem is handled below).
                diagnostics.log_model_invocation(
                    "grounded_answer_correction", provider, model_name,
                    ok=False, error=str(exc),
                )

        diagnostics.log_citation_validation(
            validated.problem,
            len(validated.citations),
            max(0, len(answer.citations) - len(validated.citations)),
            validated.recovery,
        )

        if validated.problem is None:
            payload = {
                "kind": "answer",
                "text": validated.answer,
                "citations": [
                    c.model_dump(mode="json") for c in validated.citations
                ],
                "confidence": validated.confidence,
                "insufficient_evidence": False,
            }
            if validated.recovery:
                payload["citation_recovery"] = validated.recovery
        elif validated.problem == "model_abstained":
            # The model twice judged the passages insufficient: honest
            # abstention with the model's own phrasing when available.
            payload = {
                "kind": "answer",
                "text": validated.answer or guardrails.ABSTENTION_TEXT,
                "citations": [],
                "confidence": validated.confidence,
                "insufficient_evidence": True,
            }
        else:
            # The model stood by its answer but its citations could not be
            # resolved: the CITATION is rejected, the answer is kept and
            # flagged unverified (never silently discarded).
            payload = {
                "kind": "answer",
                "text": validated.answer,
                "citations": [],
                "confidence": validated.confidence,
                "insufficient_evidence": False,
                "unverified": True,
                "validation_problem": validated.problem,
            }
        diagnostics.log_final_outcome(
            payload["kind"],
            payload["insufficient_evidence"],
            len(payload["citations"]),
            payload.get("confidence"),
        )
        return {"final_response": payload}

    # ------------------------------------------------------------------ 5
    # datetime_response (REMOVED, follow-up task 4, section 7)
    #
    # The old deterministic node answered every date/time request without a
    # model call.  It is replaced by MODEL-DRIVEN function calling: the
    # request is routed to the general_response agent, which decides to
    # invoke the registered get_current_datetime tool (core.datetime_util
    # remains the single reliable data source - see tools/datetime_tools.py;
    # a post-loop honesty check guarantees the final answer never contains
    # an invented date/time).

    # ------------------------------------------------------------------ 6
    def task_listing(state: dict) -> dict:
        """Execute the read-only list_tasks tool (C3)."""
        if cancellation.cancellation_requested():
            return _cancelled()
        _phase("Listing your tasks…")
        request = (state.get("user_request") or "").lower()
        status: TaskStatus | None = None
        if "completed" in request:
            status = TaskStatus.COMPLETED
        elif "cancelled" in request or "canceled" in request:
            status = TaskStatus.CANCELLED
        elif "open" in request:
            status = TaskStatus.OPEN

        list_tool = make_list_tasks_tool(task_repo)
        text = list_tool.invoke(
            {"status": status.value if status else "all"}
        )
        records = task_repo.list_tasks(status)
        return {
            "final_response": {
                "kind": "task_list",
                "text": text,
                "tasks": [r.model_dump(mode="json") for r in records],
            },
            "tool_results": [
                {
                    "tool": "list_tasks",
                    "input": {"status": status.value if status else "all"},
                    "result_count": len(records),
                }
            ],
        }

    # ------------------------------------------------------------------ 6
    def task_proposal(state: dict) -> dict:
        """Generate and validate a TaskProposal (C1, C5)."""
        if cancellation.cancellation_requested():
            return _cancelled()
        _phase("Preparing a task proposal…")
        try:
            proposal = gateway.structured(
                TaskProposal,
                prompts.task_proposal_messages(
                    state["user_request"],
                    _history(state),
                    date.today(),
                    _prefs(state),
                ),
                run_name="task_proposal",
                metadata={"prompt_version": prompts.PROMPT_VERSIONS["task_proposal"]},
            )
        except ModelGatewayError as exc:
            return {
                "errors": [f"Task proposal generation failed: {exc}"],
                "final_response": {
                    "kind": "clarification",
                    "text": (
                        "I could not prepare a valid task proposal from that "
                        "request. Could you rephrase it and tell me the task "
                        "title (and, optionally, the due date and priority)?\n\n"
                        f"Reason: {str(exc)[:200]}"
                    ),
                },
            }

        from core.models import new_id

        operation_id = new_id()
        return {
            "task_proposal": proposal.model_dump(mode="json"),
            "operation_id": operation_id,
            "approval_status": "pending",
            "final_response": {
                "kind": "proposal_pending",
                "text": (
                    f"{proposal.confirmation_message}\n\n"
                    "This task has NOT been saved yet. It will only be created "
                    "after you approve it."
                ),
                "proposal": proposal.model_dump(mode="json"),
                "operation_id": operation_id,
            },
        }

    # ------------------------------------------------------------------ 7
    def human_approval(state: dict) -> dict:
        """Durable pause: wait for explicit human approval (C5, E2).

        LangGraph interrupt rules: no try/except around interrupt(), no side
        effects before it, JSON-serializable payload.  Silence, timeouts,
        refreshes, and ambiguous resume values NEVER count as approval.
        """
        proposal = state.get("task_proposal") or {}
        operation_id = state.get("operation_id") or ""
        decision = interrupt(
            {
                "type": "task_approval",
                "operation_id": operation_id,
                "proposal": proposal,
                "question": (
                    "Approve creating this task? The task is NOT saved until "
                    "you explicitly approve."
                ),
            }
        )
        if decision not in ("approved", "rejected"):
            # Ambiguous input is never treated as approval.
            decision = "rejected"
        if decision == "approved":
            return {"approval_status": "approved"}
        return {
            "approval_status": "rejected",
            "final_response": {
                "kind": "task_rejected",
                "text": (
                    "The task proposal was discarded. No task has been "
                    "created. If you want, rephrase the request and I will "
                    "propose it again."
                ),
                "proposal": proposal,
                "operation_id": operation_id,
            },
        }

    # ------------------------------------------------------------------ 8
    def task_execution(state: dict) -> dict:
        """Execute create_task only after approval (C3, C5, E2).

        Idempotent: if this thread already created the task (double approve,
        resumed execution), nothing is written again.

        Cancellation is checked IMMEDIATELY before the irreversible write:
        a Stop press between approval and this node prevents the task from
        being created.  (The write itself is a single atomic INSERT guarded
        by UNIQUE(operation_id), so once it has begun it completes once.)
        """
        if state.get("task_id"):
            return {}
        # Check cancellation both from the graph state (an earlier node in
        # THIS superstep chain noticed it) and the live event (Stop clicked
        # while the approval-resume run was executing).
        if state.get("cancelled") or cancellation.cancellation_requested():
            return _cancelled()

        proposal_data = state.get("task_proposal") or {}
        try:
            proposal = TaskProposal(**proposal_data)
        except ValidationError as exc:
            return {"errors": [f"Stored proposal became invalid: {exc}"]}

        create_tool = make_create_task_tool(task_repo)
        raw = create_tool.invoke(
            {
                "title": proposal.title,
                "description": proposal.description,
                "due_date": proposal.due_date.isoformat()
                if proposal.due_date
                else None,
                "priority": proposal.priority.value,
                "operation_id": state.get("operation_id") or "",
            }
        )
        result = json.loads(raw)
        if result.get("status") == "rejected":
            return {"errors": [f"Task creation rejected: {result.get('error')}"]}

        task = result["task"]
        already = result.get("status") == "already_exists"
        return {
            "task_id": task["id"],
            "final_response": {
                "kind": "task_created",
                "text": (
                    f"Task '{task['title']}' was already created earlier; "
                    "no duplicate was made."
                    if already
                    else f"Task '{task['title']}' has been created and saved."
                ),
                "tasks": [task],
                "operation_id": state.get("operation_id"),
            },
            "tool_results": [
                {
                    "tool": "create_task",
                    "input": {"operation_id": state.get("operation_id")},
                    "result": result.get("status"),
                }
            ],
        }

    # ------------------------------------------------------------------ 9
    def _execute_tool_calls(
        response: AIMessage,
        registry,
        loop_messages: list,
        tool_trace: list[dict],
        counters: dict,
    ) -> str:
        """Execute one assistant message's tool calls; return a status.

        Status: "ok" (all calls handled), "over_budget" (the per-request
        tool budget was exhausted - every remaining call was answered with
        an explicit budget message so the conversation stays valid for
        strict OpenAI-compatible providers), or "cancelled".

        Every argument is validated INDEPENDENTLY of the model by the
        tool's Pydantic args_schema (an invalid schema raises inside
        .invoke and is fed back to the model as an error string).  Only
        registered READ-ONLY tools can ever execute here; the restricted
        create_task tool is not in the bindable set and its invocation is
        refused (follow-up task 4, section 11).
        """
        for tool_call in response.tool_calls or []:
            if cancellation.cancellation_requested():
                return "cancelled"
            name = str(tool_call.get("name") or "")
            args = dict(tool_call.get("args") or {})
            if counters["calls"] >= settings.max_total_tool_calls:
                loop_messages.append(
                    ToolMessage(
                        content=(
                            "Tool call budget exhausted. Do not call more "
                            "tools; answer the user now with the "
                            "information you already have."
                        ),
                        tool_call_id=str(tool_call.get("id") or ""),
                    )
                )
                return "over_budget"
            registered = registry.get(name)
            if registered is None or not registered.bindable:
                output = (
                    f"Error: tool '{name}' is not available. Use one of "
                    "the offered tools only."
                )
                loop_messages.append(
                    ToolMessage(
                        content=output, tool_call_id=str(tool_call.get("id") or "")
                    )
                )
                tool_trace.append(
                    {"tool": name, "input": args, "error": "unavailable tool"}
                )
                continue
            try:
                signature = (name, json.dumps(args, sort_keys=True))
            except TypeError:
                signature = (name, str(args))
            if signature in counters["seen"]:
                loop_messages.append(
                    ToolMessage(
                        content=(
                            "You already called this tool with identical "
                            "arguments. Reuse the earlier result instead "
                            "of calling it again."
                        ),
                        tool_call_id=str(tool_call.get("id") or ""),
                    )
                )
                tool_trace.append({"tool": name, "input": args,
                                   "note": "duplicate call suppressed"})
                continue
            counters["seen"].add(signature)
            counters["calls"] += 1
            _phase(f"Using {name}…")
            started = time.perf_counter()
            ok = True
            try:
                output = registered.tool.invoke(args)
            except Exception as exc:  # noqa: BLE001 - fed back to the model
                ok = False
                output = f"Tool error ({type(exc).__name__}): {exc}"
            duration_ms = round((time.perf_counter() - started) * 1000)
            if ok and name == "get_current_datetime":
                counters["datetime_used"] = True
            loop_messages.append(
                ToolMessage(
                    content=str(output), tool_call_id=str(tool_call.get("id") or "")
                )
            )
            trace_entry = {
                "tool": name,
                "input": {
                    key: (str(value)[:120] if not isinstance(value, (int, float, bool, type(None))) else value)
                    for key, value in list(args.items())[:6]
                },
                # Truncated tool output: enough to audit what the tool
                # actually returned (also surfaced to LangSmith via the
                # next call's metadata and to the UI trace) without
                # dumping whole documents into the state.
                "result": str(output)[:240],
                "duration_ms": duration_ms,
                "ok": ok,
            }
            search_provider = (getattr(registered, "meta", None) or {}).get(
                "search_provider"
            )
            if search_provider:
                # e.g. the web_search tool records which search provider
                # served the call (follow-up task 4, section 13).
                trace_entry["provider"] = search_provider
            tool_trace.append(trace_entry)
        return "ok"

    def general_response(state: dict) -> dict:
        """Model-driven tool-calling agent (follow-up task 4, sections 6-7).

        The MODEL decides when a registered tool is needed; the application
        validates every argument, executes through application-controlled
        functions only, bounds the loop (MAX_TOOL_ITERATIONS model turns,
        MAX_TOTAL_TOOL_CALLS executions), suppresses repeated identical
        calls, and preserves Stop-cancellation at every safe point.

        Conversation validity: the assistant message containing tool_calls
        is appended BEFORE its ToolMessages (required by strict
        OpenAI-compatible endpoints; the previous loop omitted it).
        """
        if cancellation.cancellation_requested():
            return _cancelled()
        _phase("Thinking…")
        request = state.get("user_request") or ""

        registry, bindable = _request_registry()
        if search_service is not None:
            search_service.reset_run_scope()
        loop_messages = prompts.general_assistant_messages(
            _prefs(state), _history(state), request, bindable_tools=bindable
        )
        tool_trace: list[dict] = []
        counters = {"calls": 0, "seen": set(), "datetime_used": False}
        response: AIMessage | None = None
        exhausted = False

        try:
            for iteration in range(settings.max_tool_iterations):
                if cancellation.cancellation_requested():
                    return {**_cancelled(), "tool_results": tool_trace}
                _phase("Thinking…")
                response = gateway.chat(
                    loop_messages,
                    tools=bindable or None,
                    run_name="agent_response",
                    metadata={
                        "prompt_version": prompts.PROMPT_VERSIONS["system_assistant"],
                        "tools_offered": _tool_names(bindable),
                        "tool_iterations_used": iteration,
                        "tool_calls_completed": counters["calls"],
                    },
                )
                if not response.tool_calls:
                    break
                # CRITICAL: the tool-call request message itself belongs in
                # the conversation before any ToolMessage (strict providers
                # reject a tool message without its assistant tool_calls).
                loop_messages.append(response)
                status = _execute_tool_calls(
                    response, registry, loop_messages, tool_trace, counters
                )
                if status == "cancelled":
                    return {**_cancelled(), "tool_results": tool_trace}
                if status == "over_budget":
                    exhausted = True
                    break
            else:
                # Every iteration was spent on tool calls and the model
                # still wanted more: the budget is exhausted.
                exhausted = True
            if exhausted:
                # Bounded shutdown: ask for the final answer with NO tools
                # bound (an explicit, useful error instead of a dead loop).
                loop_messages.append(
                    HumanMessage(
                        content=(
                            "Tool call budget exhausted: answer the user's "
                            "request now with the information you already "
                            "have, and mention briefly which part could not "
                            "be completed."
                        )
                    )
                )
                response = gateway.chat(
                    loop_messages,
                    run_name="agent_response_final",
                    metadata={
                        "tool_budget_exhausted": True,
                        "tool_calls_completed": counters["calls"],
                    },
                )
        except ModelGatewayError as exc:
            return {"errors": [f"Response generation failed: {exc}"]}

        text = str((response or AIMessage(content="")).content or "").strip()

        # ---- datetime honesty check (follow-up task 4, section 7) --------
        # A date/time-shaped question must never be answered with an
        # invented timestamp.  If the model answered without calling
        # get_current_datetime: ONE corrective retry with tools bound;
        # if it still refuses the tool, append the deterministic
        # application-clock answer (clearly marked) so no fabricated
        # date/time survives.
        datetime_shaped = guardrails.looks_like_datetime_query(request)
        if datetime_shaped and not counters["datetime_used"]:
            try:
                retry_messages = list(loop_messages)
                if text:
                    retry_messages.append(AIMessage(content=text))
                retry_messages.append(
                    HumanMessage(
                        content=(
                            "That was a question about the current date or "
                            "time, but you did not call the "
                            "get_current_datetime tool. Call it now and "
                            "answer strictly from its result - never guess "
                            "the current date or time."
                        )
                    )
                )
                retry = gateway.chat(
                    retry_messages,
                    tools=bindable or None,
                    run_name="agent_datetime_correction",
                    metadata={"datetime_honesty_retry": True},
                )
                if retry.tool_calls:
                    retry_messages.append(retry)
                    status = _execute_tool_calls(
                        retry, registry, retry_messages, tool_trace, counters
                    )
                    if status == "ok":
                        retry = gateway.chat(
                            retry_messages,
                            run_name="agent_datetime_correction_final",
                            metadata={"datetime_honesty_retry": True},
                        )
                if counters["datetime_used"]:
                    text = str(retry.content or "").strip() or text
            except ModelGatewayError:
                pass  # keep the original answer; the footer below still applies
            if not counters["datetime_used"]:
                footer = (
                    datetime_util.datetime_payload()["text"]
                    + "  (Added from the application clock: the assistant "
                    "did not call the date/time tool, so the answer above "
                    "may have contained a guessed date.)"
                )
                text = f"{text}\n\n{footer}".strip()

        return {
            "final_response": {
                "kind": "conversation",
                "text": text or "I am not sure how to respond to that.",
            },
            "tool_results": tool_trace,
        }

    # ------------------------------------------------------------------ 9b
    def fact_check_claims(state: dict) -> dict:
        """Fact-check step 1: retrieve the relevant document evidence and
        extract its verifiable factual claims (follow-up task 4, 8.4-8.5).

        Retrieval strategy (deterministic, bounded):
        - the request names a specific indexed file -> that document's
          passages (chunk order);
        - the library holds exactly one document -> that document (the
          user said "my document");
        - otherwise -> hybrid retrieval with the request, falling back to
          the most recent document's passages when retrieval finds nothing
          (a bare "fact-check my notes" must still find the notes).

        The extraction call itself is structured (ExtractedClaims) and the
        original claim wording is preserved verbatim.
        """
        if cancellation.cancellation_requested():
            return _cancelled()
        request = state.get("user_request") or ""
        tool_trace: list[dict] = []

        if (
            services.doc_repo is None
            or services.doc_repo.count_documents(("indexed",)) == 0
        ):
            return {
                "final_response": {
                    "kind": "answer",
                    "text": guardrails.EMPTY_LIBRARY_TEXT,
                    "citations": [],
                    "insufficient_evidence": True,
                    "library_empty": True,
                },
                "tool_results": [
                    {"tool": "get_document_content", "input": {},
                     "note": "library empty"}
                ],
            }

        _phase("Retrieving the relevant document passages…")
        max_chunks = max(8, settings.fact_check_max_claims * 2)
        chunks: list[RetrievedChunk] = []
        retrieval_note = "hybrid search"
        named: list = []
        if services.doc_repo is not None:
            for record in services.doc_repo.list_documents():
                if record.status != "indexed":
                    continue
                if record.filename and record.filename in request:
                    named.append(record)
        if named:
            for record in named[:2]:
                chunks.extend(
                    retriever.document_chunks(record.document_id, limit=max_chunks)
                )
            retrieval_note = f"named document(s): {', '.join(r.filename for r in named[:2])}"
        else:
            indexed = [
                r
                for r in (services.doc_repo.list_documents() if services.doc_repo else [])
                if r.status == "indexed"
            ]
            if len(indexed) == 1:
                chunks = retriever.document_chunks(
                    indexed[0].document_id, limit=max_chunks
                )
                retrieval_note = f"single document: {indexed[0].filename}"
            else:
                try:
                    chunks = retriever.search(request, top_k=max_chunks)
                except Exception as exc:  # noqa: BLE001 - infrastructure error
                    diagnostics.log_retrieval_failure("fact_check_retrieval", exc)
                    return {
                        "errors": [
                            f"Document search is currently unavailable "
                            f"({type(exc).__name__}): {str(exc)[:200]}"
                        ]
                    }
                if not chunks and indexed:
                    # "fact-check my notes" style requests can retrieve
                    # nothing by similarity - fall back to the newest doc.
                    newest = max(indexed, key=lambda r: r.uploaded_at or "")
                    chunks = retriever.document_chunks(
                        newest.document_id, limit=max_chunks
                    )
                    retrieval_note = f"fallback to newest document: {newest.filename}"

        if not chunks:
            return {
                "final_response": {
                    "kind": "fact_check_report",
                    "text": (
                        "I could not retrieve any document passages to "
                        "fact-check. Upload or index the document first, "
                        "or name the file you want verified."
                    ),
                    "report": None,
                },
                "tool_results": [
                    {"tool": "get_document_content", "input": {},
                     "note": "no passages retrieved"}
                ],
            }
        diagnostics.log_retrieval(f"fact_check ({retrieval_note})", chunks)
        tool_trace.append(
            {"tool": "get_document_content", "input": {"mode": retrieval_note},
             "result_count": len(chunks)}
        )

        _phase("Extracting factual claims…")
        try:
            extracted = gateway.structured(
                ExtractedClaims,
                prompts.factcheck_extraction_messages(
                    request, chunks, settings.fact_check_max_claims
                ),
                run_name="fact_check_extraction",
                metadata={
                    "prompt_version": prompts.PROMPT_VERSIONS[
                        "fact_check_extraction"
                    ],
                    "passages": len(chunks),
                },
            )
        except ModelGatewayError as exc:
            return {"errors": [f"Claim extraction failed: {exc}"]}

        claims = [c.model_dump(mode="json") for c in extracted.claims]
        if not claims:
            return {
                "final_response": {
                    "kind": "fact_check_report",
                    "text": (
                        "I read the relevant passages and did not find "
                        "verifiable factual claims (opinions, questions, "
                        "or personal preferences are not treated as "
                        "objective facts). Nothing to fact-check."
                    ),
                    "report": None,
                },
                "tool_results": tool_trace,
            }
        return {
            "factcheck_claims": claims,
            "retrieved_documents": [c.model_dump(mode="json") for c in chunks],
            "tool_results": tool_trace,
        }

    # ------------------------------------------------------------------ 9c
    def fact_check_verification(state: dict) -> dict:
        """Fact-check steps 2-5 (follow-up task 4, sections 8.4-8.8).

        1. Compare every claim against the model's general knowledge
           (structured ClaimAssessmentBatch; opinions/ambiguities get their
           own statuses - never a forced true/false).
        2. For claims the model flags as needing independent evidence (and
           only those, bounded by FACT_CHECK_MAX_WEB_QUERIES), retrieve
           external evidence through the web-search service - actual
           sources only, snippets are evidence, never instructions.
        3. Produce the final FactCheckReport with the three evidence
           layers kept distinct (document / model knowledge / external).
        4. VALIDATE the findings application-side: external source URLs
           that were not actually retrieved are stripped (the model can
           never smuggle an invented citation past this check), and each
           finding's document evidence is cross-checked with the
           deterministic claim comparator.
        """
        if cancellation.cancellation_requested():
            return _cancelled()
        claims = state.get("factcheck_claims") or []
        if not claims:
            return {}
        request = state.get("user_request") or ""
        tool_trace: list[dict] = list(state.get("tool_results") or [])

        _phase("Comparing claims against known information…")
        try:
            batch = gateway.structured(
                ClaimAssessmentBatch,
                prompts.factcheck_assessment_messages(claims),
                run_name="fact_check_assessment",
                metadata={
                    "prompt_version": prompts.PROMPT_VERSIONS[
                        "fact_check_assessment"
                    ],
                    "claims": len(claims),
                },
            )
        except ModelGatewayError as exc:
            return {"errors": [f"Claim assessment failed: {exc}"]}

        # Match assessments back to claims (verbatim first, positional
        # fallback if the model lightly reworded a claim_text).
        assessment_by_text = {a.claim_text: a for a in batch.assessments}
        matched: list[dict] = []
        for index, claim in enumerate(claims):
            assessment = assessment_by_text.get(claim.get("claim_text", ""))
            if assessment is None and index < len(batch.assessments):
                candidate = batch.assessments[index]
                if candidate.claim_text not in assessment_by_text.values():
                    assessment = candidate
            if assessment is not None:
                matched.append(
                    {
                        "claim": claim,
                        "assessment": assessment.model_dump(mode="json"),
                    }
                )
        if not matched:
            return {
                "errors": [
                    "The fact-check assessment could not be matched back "
                    "to the extracted claims; please rephrase the request."
                ]
            }

        # ---- external evidence (bounded, needs-flagged claims only) --------
        web_results: dict[int, list[dict]] = {}
        search_notes: list[str] = []
        if search_service is not None and search_service.available:
            search_service.reset_run_scope()
            budget = settings.fact_check_max_web_queries
            for entry in matched:
                if budget <= 0:
                    break
                if not entry["assessment"].get("needs_external_verification"):
                    continue
                claim_text = entry["claim"].get("claim_text", "")[:200]
                _phase("Searching the web for independent evidence…")
                try:
                    results = search_service.search(
                        claim_text, max_results=settings.web_search_max_results
                    )
                except Exception as exc:  # noqa: BLE001 - evidence step
                    search_notes.append(f"web search failed: {exc}")
                    continue
                budget -= 1
                web_results[matched.index(entry) + 1] = [
                    r.to_dict() for r in results
                ]
                tool_trace.append(
                    {
                        "tool": "web_search",
                        "input": {"query": claim_text},
                        "result_count": len(results),
                        "provider": search_service.provider_name,
                    }
                )
        elif any(
            entry["assessment"].get("needs_external_verification")
            for entry in matched
        ):
            search_notes.append(
                "web search is not configured; claims needing independent "
                "verification are marked unverified where appropriate"
            )

        _phase("Validating findings…")
        try:
            report = gateway.structured(
                FactCheckReport,
                prompts.factcheck_report_messages(
                    [entry["claim"] for entry in matched],
                    [entry["assessment"] for entry in matched],
                    web_results,
                ),
                run_name="fact_check_report",
                metadata={
                    "prompt_version": prompts.PROMPT_VERSIONS[
                        "fact_check_report"
                    ],
                    "claims": len(matched),
                    "web_queries": len(web_results),
                },
            )
        except ModelGatewayError as exc:
            return {"errors": [f"Fact-check report generation failed: {exc}"]}

        # ---- application-side validation (sections 8.7, 9.2) ---------------
        retrieved_urls = {
            result["url"]
            for results in web_results.values()
            for result in results
        }
        from tools.verify_claims_tool import compare_claim_to_evidence

        validated_findings = []
        for finding in report.findings:
            sources = []
            dropped = 0
            for source in finding.external_sources:
                if source.url in retrieved_urls:
                    sources.append(source)
                else:
                    dropped += 1
            limitations = finding.limitations
            if dropped:
                limitations = (
                    f"{limitations + '; ' if limitations else ''}"
                    f"{dropped} cited source(s) were not actually "
                    "retrieved and were removed"
                )
            # If EVERY external source was stripped, the finding no longer
            # rests on independent evidence - downgrade the declared
            # verification method so the report label matches reality.
            if dropped and not sources and "external" in (
                finding.verification_method or ""
            ):
                limitations = (
                    f"{limitations + '; ' if limitations else ''}"
                    "verification downgraded to model knowledge (no "
                    "independently retrieved source survived validation)"
                )
                finding = finding.model_copy(
                    update={"verification_method": "model_knowledge"}
                )
            # Deterministic cross-check of the document evidence against
            # the ORIGINAL claim text (a signal recorded in the trace, not
            # a verdict override).
            claim_source = None
            for entry in matched:
                if entry["claim"].get("claim_text") == finding.original_claim:
                    claim_source = entry["claim"]
                    break
            comparison_note = None
            if claim_source and finding.document_evidence:
                comparison = compare_claim_to_evidence(
                    finding.original_claim, finding.document_evidence
                )
                if comparison["signal"] == "insufficient_overlap":
                    comparison_note = (
                        "the quoted document evidence has low lexical "
                        "overlap with the original claim - verify the "
                        "source passage"
                    )
            if comparison_note:
                limitations = (
                    f"{limitations + '; ' if limitations else ''}"
                    f"{comparison_note}"
                )
            validated_findings.append(
                finding.model_copy(
                    update={
                        "external_sources": sources,
                        "limitations": limitations,
                    }
                ).model_dump(mode="json")
            )

        diagnostics.log_factcheck_validation(
            claims=len(matched),
            stripped_sources=sum(
                1 for f in report.findings
                if any(s.url not in retrieved_urls for s in f.external_sources)
            ),
            notes=search_notes,
        )
        status_counts: dict[str, int] = {}
        for finding in validated_findings:
            status_counts[finding["status"]] = (
                status_counts.get(finding["status"], 0) + 1
            )
        summary = report.summary or (
            ", ".join(
                f"{count} {status}"
                for status, count in sorted(status_counts.items())
            )
            or "no findings"
        )
        text = prompts.factcheck_summary_text(
            request, validated_findings, search_notes
        )
        return {
            "final_response": {
                "kind": "fact_check_report",
                "text": text,
                "report": {
                    "findings": validated_findings,
                    "summary": summary,
                    "search_notes": search_notes,
                },
            },
            "tool_results": tool_trace,
        }

    # ------------------------------------------------------------------ 10
    def final_response(state: dict) -> dict:
        """Prepare the result for the UI and append the assistant message.

        When the run was cancelled (Stop pressed), the assistant message is
        an explicit stopped marker - the transcript stays coherent
        (user message + stopped marker, no dangling unanswered request) and
        the composer is re-enabled.
        """
        stopped = bool(state.get("cancelled")) or cancellation.cancellation_requested()
        payload = dict(state.get("final_response") or {})
        errors = state.get("errors") or []
        if stopped:
            payload = {
                "kind": "stopped",
                "text": STOPPED_TEXT,
                "stopped": True,
            }
        elif errors and not payload.get("text"):
            payload = {
                "kind": "error",
                "text": "Sorry, something went wrong while handling that request:\n"
                + "\n".join(f"- {e}" for e in errors[:3]),
                "error": errors[0],
            }
        elif errors:
            payload["errors"] = errors[:3]

        thread_id = state.get("thread_id") or ""
        if thread_id:
            title = (state.get("user_request") or "")[:60] or None
            try:
                services.conversation_repo.ensure_conversation(thread_id, title)
                services.conversation_repo.touch(thread_id)
            except Exception:  # noqa: BLE001 - UI bookkeeping must not break the graph
                pass

        message = AIMessage(
            content=payload.get("text", ""),
            additional_kwargs={"pkta_payload": payload},
        )
        return {"messages": [message], "final_response": payload}

    return {
        "input_guardrail": input_guardrail,
        "intent_router": intent_router,
        "document_retrieval": document_retrieval,
        "grounded_answer": grounded_answer,
        # datetime_response removed (follow-up task 4, section 7): the
        # model now invokes the get_current_datetime tool itself inside
        # the general_response agent loop.
        "task_listing": task_listing,
        "task_proposal": task_proposal,
        "human_approval": human_approval,
        "task_execution": task_execution,
        "fact_check_claims": fact_check_claims,
        "fact_check_verification": fact_check_verification,
        "general_response": general_response,
        "final_response": final_response,
    }
