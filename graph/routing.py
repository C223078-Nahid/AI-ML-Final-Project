"""Conditional routing functions for the workflow graph (C7).

Cancellation routing (follow-up task, section 7): every route function
first checks ``state["cancelled"]``.  When the user pressed Stop, whatever
node noticed it set the flag and short-circuited; every subsequent edge
leads to ``final_response`` which emits the "stopped" marker.  The graph
therefore always completes its normal path - the checkpointer never sees a
half-finished superstep, no tool or task write happens after cancellation.
"""

from __future__ import annotations

INTENT_ROUTES: dict[str, str] = {
    "document_question": "document_retrieval",
    "list_tasks": "task_listing",
    "create_task": "task_proposal",
    "general_knowledge": "general_response",
    "general_conversation": "general_response",
    # Follow-up task 4, section 7: date/time requests are no longer forced
    # through a deterministic graph node - they go to the model-driven
    # tool-calling agent, which decides to invoke get_current_datetime.
    "current_datetime": "general_response",
    # Follow-up task 4, section 8: the explicit fact-check workflow.
    "fact_check": "fact_check_claims",
    "unsupported_request": "final_response",
}


def route_after_guardrail(state: dict) -> str:
    """Refusals and validation errors skip straight to the final response."""
    if state.get("cancelled"):
        return "final_response"
    if state.get("errors"):
        return "final_response"
    if (state.get("final_response") or {}).get("kind") == "refusal":
        return "final_response"
    return "intent_router"


def route_after_intent(state: dict) -> str:
    if state.get("cancelled"):
        return "final_response"
    if state.get("errors"):
        return "final_response"
    if (state.get("final_response") or {}).get("kind") == "refusal":
        return "final_response"
    intent = (state.get("intent") or {}).get("intent")
    # Unknown or malformed classification results never fall into the RAG
    # branch by default: general conversation is the safe default route.
    return INTENT_ROUTES.get(intent, "general_response")


def route_after_factcheck_claims(state: dict) -> str:
    """After the claim-extraction node of the fact-check workflow.

    - ``final_response`` when cancelled, on errors, or when extraction
      already produced a complete response payload (empty library, no
      verifiable claims found);
    - ``fact_check_verification`` in the normal case.
    """
    if state.get("cancelled"):
        return "final_response"
    if state.get("errors"):
        return "final_response"
    if (state.get("final_response") or {}).get("kind"):
        return "final_response"
    if not state.get("factcheck_claims"):
        return "final_response"
    return "fact_check_verification"


def route_after_retrieval(state: dict) -> str:
    """After the retrieval node.

    Four destinations:
    - ``final_response`` when the run was cancelled (Stop pressed);
    - ``final_response`` on errors (infrastructure failures are handled as
      errors, never silently as "no evidence");
    - ``final_response`` when the retrieval node already produced a complete
      response payload (e.g. the library is empty: a distinct message, not
      a generic abstention);
    - ``grounded_answer`` in the normal case.
    """
    if state.get("cancelled"):
        return "final_response"
    if state.get("errors"):
        return "final_response"
    if (state.get("final_response") or {}).get("kind"):
        return "final_response"
    return "grounded_answer"


def route_after_proposal(state: dict) -> str:
    if state.get("cancelled"):
        return "final_response"
    if state.get("errors") or not state.get("task_proposal"):
        return "final_response"
    return "human_approval"


def route_after_approval(state: dict) -> str:
    if state.get("cancelled"):
        return "final_response"
    if state.get("approval_status") == "approved":
        return "task_execution"
    return "final_response"
