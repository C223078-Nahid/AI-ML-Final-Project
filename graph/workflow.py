"""Graph assembly (concept C7).

The compiled StateGraph is the actual controller of application execution
(follow-up task 4, section 12):

    input_guardrail -> intent_router -> one of
        document_retrieval -> grounded_answer          (citation-grounded RAG)
        general_response                                 (model-driven tool
                                                         loop: documents,
                                                         date/time, web
                                                         search, tasks)
        fact_check_claims -> fact_check_verification    (explicit fact-check)
        task_listing                                     (read-only)
        task_proposal -> human_approval -> task_execution (durable interrupt)
    -> final_response

The task-creation branch pauses at human_approval via interrupt() and
resumes with Command(resume=...).  The deterministic datetime branch was
REMOVED (follow-up task 4, section 7): current date/time questions now go
to the general_response agent, where the model invokes the registered
get_current_datetime tool through function calling.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import GraphServices, make_nodes
from graph.routing import (
    route_after_approval,
    route_after_factcheck_claims,
    route_after_guardrail,
    route_after_intent,
    route_after_proposal,
    route_after_retrieval,
)
from graph.state import WorkflowState


def build_workflow(
    services: GraphServices, checkpointer: BaseCheckpointSaver
):
    """Compile the workflow graph with a persistent checkpointer."""
    builder = StateGraph(WorkflowState)
    for name, node in make_nodes(services).items():
        builder.add_node(name, node)

    builder.add_edge(START, "input_guardrail")

    builder.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"intent_router": "intent_router", "final_response": "final_response"},
    )
    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "document_retrieval": "document_retrieval",
            "task_listing": "task_listing",
            "task_proposal": "task_proposal",
            "general_response": "general_response",
            "fact_check_claims": "fact_check_claims",
            "final_response": "final_response",
        },
    )
    builder.add_conditional_edges(
        "document_retrieval",
        route_after_retrieval,
        {"grounded_answer": "grounded_answer", "final_response": "final_response"},
    )
    builder.add_edge("grounded_answer", "final_response")
    builder.add_edge("task_listing", "final_response")

    # Explicit fact-check workflow (follow-up task 4, section 8.4):
    # claim extraction -> verification (with bounded external evidence)
    # -> final report.
    builder.add_conditional_edges(
        "fact_check_claims",
        route_after_factcheck_claims,
        {
            "fact_check_verification": "fact_check_verification",
            "final_response": "final_response",
        },
    )
    builder.add_edge("fact_check_verification", "final_response")

    builder.add_conditional_edges(
        "task_proposal",
        route_after_proposal,
        {"human_approval": "human_approval", "final_response": "final_response"},
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"task_execution": "task_execution", "final_response": "final_response"},
    )
    builder.add_edge("task_execution", "final_response")

    builder.add_edge("general_response", "final_response")
    builder.add_edge("final_response", END)

    return builder.compile(checkpointer=checkpointer)
