"""Seed a fact-check conversation into the running app's data directory.

Uses the REAL application container (real embeddings, real Qdrant store,
real LangGraph workflow with SQLite checkpointer) but a scripted gateway
double - the same pattern as the offline test suite - so the fact-check
report UI can be verified in the browser without real model credentials.

Run from the project root:  python scripts/seed_factcheck_conversation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from langchain_core.messages import AIMessage  # noqa: E402

from core.models import (  # noqa: E402
    ClaimAssessment,
    ClaimAssessmentBatch,
    ExtractedClaim,
    ExtractedClaims,
    ExternalSource,
    FactCheckReport,
    FactCheckResult,
    Intent,
    IntentResult,
    VerificationStatus,
)
from tests.fakes import FakeGateway  # noqa: E402


def main() -> None:
    from app import build_container
    from config.settings import get_settings
    from ui.graph_service import send_message

    settings = get_settings()
    container = build_container(settings)

    docs = container.doc_repo.list_documents()
    chemistry = next(
        (d for d in docs if d.filename.startswith("Chemistry_Notes")), None
    )
    if chemistry is None:
        raise SystemExit(
            "Chemistry_Notes.txt is not indexed yet - upload it first."
        )
    document_id = chemistry.document_id

    claims = ExtractedClaims(
        claims=[
            ExtractedClaim(
                claim_text="The chemical symbol for gold is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                chunk_id=f"{document_id}:0",
                context="Chemistry notes on element symbols.",
            ),
            ExtractedClaim(
                claim_text="The chemical symbol for silver is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                chunk_id=f"{document_id}:0",
                context="Chemistry notes on element symbols.",
            ),
            ExtractedClaim(
                claim_text="Water has the chemical formula H2O.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                chunk_id=f"{document_id}:0",
                context="Chemistry notes on formulas.",
            ),
        ]
    )
    assessments = ClaimAssessmentBatch(
        assessments=[
            ClaimAssessment(
                claim_text="The chemical symbol for gold is Ag.",
                status=VerificationStatus.INCORRECT,
                confidence=0.97,
                corrected_statement="The chemical symbol for gold is Au.",
                explanation=(
                    "Ag is the chemical symbol for silver, not gold. "
                    "Gold's chemical symbol is Au (from the Latin aurum)."
                ),
                needs_external_verification=True,
            ),
            ClaimAssessment(
                claim_text="The chemical symbol for silver is Ag.",
                status=VerificationStatus.CORRECT,
                confidence=0.98,
                explanation="Silver's chemical symbol is indeed Ag.",
            ),
            ClaimAssessment(
                claim_text="Water has the chemical formula H2O.",
                status=VerificationStatus.CORRECT,
                confidence=0.99,
                explanation="Water is H2O: two hydrogen atoms, one oxygen.",
            ),
        ]
    )
    report = FactCheckReport(
        findings=[
            FactCheckResult(
                original_claim="The chemical symbol for gold is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                status=VerificationStatus.INCORRECT,
                confidence=0.97,
                corrected_statement="The chemical symbol for gold is Au.",
                explanation=(
                    "Ag is the chemical symbol for silver (from the Latin "
                    "argentum), not gold. The document has assigned "
                    "silver's chemical symbol to gold."
                ),
                document_evidence=(
                    "The chemical symbol for gold is Ag. "
                    "(Chemistry_Notes.txt)"
                ),
                verification_method="model_knowledge_and_external",
                external_sources=[
                    ExternalSource(
                        title="Gold - Element information (Royal Society "
                              "of Chemistry)",
                        url="https://www.rsc.org/periodic-table/element/"
                            "79/gold",
                        domain="rsc.org",
                        excerpt=(
                            "Gold is a chemical element with symbol Au."
                        ),
                        retrieval_date="2026-11-25T00:00:00+00:00",
                    )
                ],
            ),
            FactCheckResult(
                original_claim="The chemical symbol for silver is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                status=VerificationStatus.CORRECT,
                confidence=0.98,
                explanation="Silver's chemical symbol is indeed Ag.",
                document_evidence=(
                    "The chemical symbol for silver is Ag. "
                    "(Chemistry_Notes.txt)"
                ),
                verification_method="model_knowledge",
            ),
            FactCheckResult(
                original_claim="Water has the chemical formula H2O.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                status=VerificationStatus.CORRECT,
                confidence=0.99,
                explanation="Water is H2O.",
                document_evidence=(
                    "Water has the chemical formula H2O. "
                    "(Chemistry_Notes.txt)"
                ),
                verification_method="model_knowledge",
            ),
        ],
        summary="1 incorrect, 2 correct",
    )

    gateway = FakeGateway(
        intent=IntentResult(intent=Intent.FACT_CHECK, confidence=0.95),
        structured_map={
            ExtractedClaims: claims,
            ClaimAssessmentBatch: assessments,
            FactCheckReport: report,
        },
    )
    container.graph  # container built; swap the gateway via services
    # The graph was built with the real gateway; rebuild the graph nodes
    # with the scripted gateway but the SAME checkpointer/repositories.
    from graph.nodes import GraphServices  # noqa: E402
    from graph.persistence import create_checkpointer  # noqa: E402
    from graph.workflow import build_workflow  # noqa: E402

    services = GraphServices(
        gateway=gateway,
        retriever=container.retriever,
        task_repo=container.task_repo,
        memory_repo=container.memory_repo,
        conversation_repo=container.conversation_repo,
        settings=settings,
        doc_repo=container.doc_repo,
        tool_registry=container.tool_registry,
        web_search=container.web_search,
    )
    checkpointer = create_checkpointer(settings.checkpoint_db_path)
    seeded_graph = build_workflow(services, checkpointer)
    original_graph = container.graph
    container.graph = seeded_graph

    thread_id = "seed-factcheck-demo"
    request = (
        "Fact-check the chemical information in my uploaded document."
    )
    result = send_message(container, thread_id, request)
    payload = result.get("final_response") or {}
    print("seeded thread:", thread_id)
    print("kind:", payload.get("kind"))
    print("findings:", len((payload.get("report") or {}).get("findings") or []))
    assert payload.get("kind") == "fact_check_report"
    assert len((payload.get("report") or {}).get("findings") or []) == 3
    container.graph = original_graph
    print("OK - open the app and click the chat to view the report.")


if __name__ == "__main__":
    main()
