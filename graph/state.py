"""Typed graph state for the LangGraph workflow (concept C7).

Notes on state design:
- ``messages`` uses the ``add_messages`` reducer so conversation memory
  accumulates across turns within a thread.
- All other keys are replaced (not appended) by the node that owns them, and
  ``input_guardrail`` resets them at the start of every new user message, so
  state never leaks from one turn into the next.
- Everything stored here is JSON-serializable (or a LangChain message), so
  the SQLite checkpointer can persist it.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    # Conversation memory (accumulates across turns in a thread)
    messages: Annotated[list[AnyMessage], add_messages]

    # Per-turn fields (reset by input_guardrail on every new message)
    thread_id: str
    user_request: str
    intent: dict | None                    # IntentResult.model_dump()
    retrieved_documents: list[dict]        # list[RetrievedChunk.model_dump()]
    retrieval_query: str                   # context-resolved standalone query
    task_proposal: dict | None             # TaskProposal.model_dump(mode="json")
    operation_id: str | None
    approval_status: str                   # pending | approved | rejected | not_required
    task_id: str | None
    tool_results: list[dict]
    preferences: list[dict]                # long-term memory snapshot
    # Claims extracted for the explicit fact-check workflow (follow-up
    # task 4, section 8); produced by fact_check_claims, consumed by
    # fact_check_verification, reset by input_guardrail.
    factcheck_claims: list[dict] | None
    final_response: dict                   # payload rendered by the UI
    errors: list[str]
    cancelled: bool                        # Stop pressed: route to final_response
