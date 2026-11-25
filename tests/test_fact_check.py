"""Follow-up task 4, section 8: document cross-referencing and
fact-checking.

The synthetic test document and expected findings come directly from the
task (section 14 D):

    Claim                              Expected assessment
    ----------------------------------------------------
    Gold has the symbol Ag             incorrect (gold is Au; Ag = silver)
    Silver has the symbol Ag           correct
    Water has the formula H2O          correct

Plus: source-passage citation, opinion handling, invented-URL stripping,
evidence-layer distinction, and the fact that ordinary document questions
never trigger the expensive workflow.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from core.models import (
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
from tests.fakes import FakeGateway, tool_call_message

CHEMISTRY_TXT = (
    b"The chemical symbol for gold is Ag.\n\n"
    b"The chemical symbol for silver is Ag.\n\n"
    b"Water has the chemical formula H2O.\n"
)

FACT_CHECK_REQUEST = (
    "Fact-check the chemical information in my uploaded document."
)


@pytest.fixture
def chemistry_doc(ingestion, doc_repo):
    result = ingestion.ingest_file("Chemistry_Notes.txt", CHEMISTRY_TXT)
    assert result.status == "indexed"
    return doc_repo.list_documents()[0]


def _build(services, settings):
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    return build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )


def _ask(graph, thread_id, question):
    from langchain_core.messages import HumanMessage

    return graph.invoke(
        {
            "user_request": question,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=question)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def _extraction(document_id: str) -> ExtractedClaims:
    return ExtractedClaims(
        claims=[
            ExtractedClaim(
                claim_text="The chemical symbol for gold is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                page_number=None,
                section=None,
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


def _assessment_batch() -> ClaimAssessmentBatch:
    return ClaimAssessmentBatch(
        assessments=[
            ClaimAssessment(
                claim_text="The chemical symbol for gold is Ag.",
                status=VerificationStatus.INCORRECT,
                confidence=0.97,
                corrected_statement="The chemical symbol for gold is Au.",
                explanation="Ag is the chemical symbol for silver, not gold.",
                needs_external_verification=True,
            ),
            ClaimAssessment(
                claim_text="The chemical symbol for silver is Ag.",
                status=VerificationStatus.CORRECT,
                confidence=0.97,
                corrected_statement="",
                explanation="Silver's symbol is indeed Ag.",
                needs_external_verification=False,
            ),
            ClaimAssessment(
                claim_text="Water has the chemical formula H2O.",
                status=VerificationStatus.CORRECT,
                confidence=0.99,
                corrected_statement="",
                explanation="Water is H2O.",
                needs_external_verification=False,
            ),
        ]
    )


def _report(document_id: str, *, invented_source: bool = False) -> FactCheckReport:
    sources = [
        ExternalSource(
            title="Gold - Element information",
            url="https://www.rsc.org/periodic-table/element/79/gold",
            domain="rsc.org",
            excerpt="Gold is a chemical element with symbol Au.",
            publication_date=None,
            retrieval_date="2026-11-25T00:00:00+00:00",
        )
    ]
    if invented_source:
        sources.append(
            ExternalSource(
                title="Totally Real Reference",
                url="https://example.invalid/made-up",
                domain="example.invalid",
                excerpt="An invented citation.",
            )
        )
    return FactCheckReport(
        findings=[
            FactCheckResult(
                original_claim="The chemical symbol for gold is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                status=VerificationStatus.INCORRECT,
                confidence=0.97,
                corrected_statement="The chemical symbol for gold is Au.",
                explanation="Ag is the chemical symbol for silver, not gold.",
                document_evidence="The chemical symbol for gold is Ag. (Chemistry_Notes.txt)",
                verification_method="model_knowledge_and_external",
                external_sources=sources,
            ),
            FactCheckResult(
                original_claim="The chemical symbol for silver is Ag.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                status=VerificationStatus.CORRECT,
                confidence=0.97,
                corrected_statement="",
                explanation="Silver's symbol is indeed Ag.",
                document_evidence="The chemical symbol for silver is Ag.",
                verification_method="model_knowledge",
            ),
            FactCheckResult(
                original_claim="Water has the chemical formula H2O.",
                document_id=document_id,
                filename="Chemistry_Notes.txt",
                status=VerificationStatus.CORRECT,
                confidence=0.99,
                corrected_statement="",
                explanation="Water is H2O.",
                document_evidence="Water has the chemical formula H2O.",
                verification_method="model_knowledge",
            ),
        ],
        summary="1 incorrect, 2 correct",
    )


def _factcheck_gateway(document_id, *, invented_source=False):
    return FakeGateway(
        intent=IntentResult(intent=Intent.FACT_CHECK, confidence=0.95),
        structured_map={
            ExtractedClaims: _extraction(document_id),
            ClaimAssessmentBatch: _assessment_batch(),
            FactCheckReport: _report(
                document_id, invented_source=invented_source
            ),
        },
    )


# ---------------------------------------------------------------------------
# The task's expected scenario
# ---------------------------------------------------------------------------
def test_chemical_fact_check(services, settings, chemistry_doc):
    """The exact scenario from the task: three claims, three expected
    assessments, gold flagged INCORRECT with the Au/Ag explanation."""
    services.gateway = _factcheck_gateway(chemistry_doc.document_id)
    graph = _build(services, settings)
    result = _ask(graph, "fc-1", FACT_CHECK_REQUEST)
    payload = result["final_response"]

    assert payload["kind"] == "fact_check_report"
    findings = payload["report"]["findings"]
    assert len(findings) == 3

    by_claim = {f["original_claim"]: f for f in findings}
    gold = by_claim["The chemical symbol for gold is Ag."]
    silver = by_claim["The chemical symbol for silver is Ag."]
    water = by_claim["Water has the chemical formula H2O."]

    # Expected assessments from the task's table.
    assert gold["status"] == "incorrect"
    assert silver["status"] == "correct"
    assert water["status"] == "correct"
    # The gold finding explains that gold is Au and Ag represents silver.
    assert "Au" in gold["corrected_statement"]
    assert "silver" in gold["explanation"].lower()

    # Each finding cites the correct document passage.
    for finding in findings:
        assert finding["filename"] == "Chemistry_Notes.txt"
        assert finding["document_id"] == chemistry_doc.document_id

    # The workflow really ran the three structured steps.
    assert services.gateway.structured_calls == [
        "IntentResult", "ExtractedClaims", "ClaimAssessmentBatch",
        "FactCheckReport",
    ]


def test_document_passages_reached_the_extraction_call(
    services, settings, chemistry_doc
):
    """The extraction model call actually contained the document passages
    (not merely the request)."""
    services.gateway = _factcheck_gateway(chemistry_doc.document_id)
    graph = _build(services, settings)
    _ask(graph, "fc-2", FACT_CHECK_REQUEST)

    # The passages reached a model call AND were wrapped as untrusted
    # content in that same message (prompt-injection protection).
    assert any(
        "The chemical symbol for gold is Ag." in str(m.content)
        and "UNTRUSTED" in str(m.content).upper()
        for messages, _with_tools in services.gateway.seen_messages
        for m in messages
    ), "the retrieved passages never reached the model as untrusted content"


def test_web_search_used_for_flagged_claims(services, settings, chemistry_doc):
    """The gold claim was flagged needs_external_verification: external
    evidence is retrieved through the web-search service (scripted
    provider), and the finding reports the retrieved source."""
    from core.web_search import WebSearchService
    from tests.test_web_search import FakeTavily

    services.web_search = WebSearchService(FakeTavily())
    services.gateway = _factcheck_gateway(chemistry_doc.document_id)
    graph = _build(services, settings)
    result = _ask(graph, "fc-3", FACT_CHECK_REQUEST)

    payload = result["final_response"]
    findings = payload["report"]["findings"]
    gold = next(
        f for f in findings if "gold is Ag" in f["original_claim"]
    )
    assert gold["status"] == "incorrect"
    # The retrieved rsc.org source is cited with real metadata.
    assert any(
        source["url"].startswith("https://www.rsc.org")
        for source in gold["external_sources"]
    )
    # The trace records the web search run through the service.
    assert any(
        t.get("tool") == "web_search" for t in result.get("tool_results") or []
    )


def test_invented_external_sources_are_stripped(
    services, settings, chemistry_doc
):
    """Section 8.7: the model can never smuggle an invented citation past
    the application-side validation - URLs not actually retrieved are
    removed and noted as a limitation."""
    from core.web_search import WebSearchService
    from tests.test_web_search import FakeTavily

    services.web_search = WebSearchService(FakeTavily())
    services.gateway = _factcheck_gateway(
        chemistry_doc.document_id, invented_source=True
    )
    graph = _build(services, settings)
    result = _ask(graph, "fc-4", FACT_CHECK_REQUEST)
    findings = result["final_response"]["report"]["findings"]
    gold = next(
        f for f in findings if "gold is Ag" in f["original_claim"]
    )
    urls = [s["url"] for s in gold["external_sources"]]
    assert "https://example.invalid/made-up" not in urls
    assert urls == ["https://www.rsc.org/periodic-table/element/79/gold"]
    assert "removed" in gold["limitations"]


def test_web_search_unavailable_marks_unverified(
    services, settings, chemistry_doc
):
    """Without web search configured, a flagged claim degrades gracefully:
    no crash, a note explains the limitation."""
    services.gateway = _factcheck_gateway(chemistry_doc.document_id)
    graph = _build(services, settings)
    result = _ask(graph, "fc-5", FACT_CHECK_REQUEST)
    payload = result["final_response"]
    assert payload["kind"] == "fact_check_report"
    assert any(
        "web search is not configured" in note
        for note in payload["report"].get("search_notes") or []
    )


def test_fact_check_safety_net_overrides_misrouting(
    services, settings, chemistry_doc
):
    """Section 8.3: an explicit verification request reaches the fact-check
    workflow even when the classifier mislabels it document_question."""
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.DOCUMENT_QUESTION, confidence=0.9),
        structured_map={
            ExtractedClaims: _extraction(chemistry_doc.document_id),
            ClaimAssessmentBatch: _assessment_batch(),
            FactCheckReport: _report(chemistry_doc.document_id),
        },
    )
    graph = _build(services, settings)
    result = _ask(graph, "fc-6", FACT_CHECK_REQUEST)
    assert result["intent"]["intent"] == "fact_check"
    assert result["final_response"]["kind"] == "fact_check_report"


def test_ordinary_document_question_never_fact_checks(
    graph, gateway, ingestion, doc_repo
):
    result = ingestion.ingest_file(
        "plan.txt", b"Qdrant was selected as the vector database."
    )
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    """Ordinary document questions do NOT pay the fact-check cost: no
    extraction/assessment calls run."""
    from core.models import Citation, GroundedAnswer

    gateway.intent = IntentResult(intent=Intent.DOCUMENT_QUESTION,
                                  confidence=0.95)
    gateway.grounded = GroundedAnswer(
        answer="The plan says Qdrant.",
        citations=[Citation(
            document_id=document.document_id, filename="plan.txt",
            chunk_id="S1", supporting_passage="Qdrant was selected.",
        )],
        confidence=0.9,
    )
    result = _ask(graph, "fc-7", "What does my document say about the vector database?")
    assert result["final_response"]["kind"] == "answer"
    assert "ExtractedClaims" not in gateway.structured_calls
    assert "ClaimAssessmentBatch" not in gateway.structured_calls


def test_empty_library_gets_honest_message(services, settings):
    """Fact-check with no documents indexed: a distinct, honest answer."""
    services.gateway = FakeGateway(
        intent=IntentResult(intent=Intent.FACT_CHECK, confidence=0.9),
    )
    graph = _build(services, settings)
    result = _ask(graph, "fc-8", FACT_CHECK_REQUEST)
    payload = result["final_response"]
    assert payload["kind"] == "answer"
    assert payload.get("library_empty") is True


def test_opinion_claims_get_their_own_category(services, settings,
                                               chemistry_doc):
    """Section 8.5-8.6: opinions are not forced into true/false - the
    assessment schema carries the opinion status through to the report."""
    document_id = chemistry_doc.document_id
    gateway = FakeGateway(
        intent=IntentResult(intent=Intent.FACT_CHECK, confidence=0.95),
        structured_map={
            ExtractedClaims: ExtractedClaims(
                claims=[
                    ExtractedClaim(
                        claim_text="The best programming language is Python.",
                        document_id=document_id,
                        filename="Chemistry_Notes.txt",
                        chunk_id=f"{document_id}:0",
                    )
                ]
            ),
            ClaimAssessmentBatch: ClaimAssessmentBatch(
                assessments=[
                    ClaimAssessment(
                        claim_text="The best programming language is Python.",
                        status=VerificationStatus.OPINION,
                        confidence=0.9,
                        explanation=(
                            "A subjective preference, not an objective "
                            "factual claim."
                        ),
                    )
                ]
            ),
            FactCheckReport: FactCheckReport(
                findings=[
                    FactCheckResult(
                        original_claim=(
                            "The best programming language is Python."
                        ),
                        document_id=document_id,
                        filename="Chemistry_Notes.txt",
                        status=VerificationStatus.OPINION,
                        confidence=0.9,
                        explanation="A subjective preference.",
                        verification_method="model_knowledge",
                    )
                ],
                summary="1 opinion",
            ),
        },
    )
    services.gateway = gateway
    graph = _build(services, settings)
    result = _ask(graph, "fc-9", "Fact-check my document.")
    findings = result["final_response"]["report"]["findings"]
    assert findings[0]["status"] == "opinion"


def test_verification_method_downgraded_when_all_sources_stripped(
    services, settings, chemistry_doc
):
    """If the model claims external verification but every cited URL fails
    validation (web search not configured), the finding's declared method
    is downgraded to model knowledge and the limitation says so."""
    services.gateway = _factcheck_gateway(
        chemistry_doc.document_id, invented_source=True
    )
    graph = _build(services, settings)
    result = _ask(graph, "fc-10", FACT_CHECK_REQUEST)
    findings = result["final_response"]["report"]["findings"]
    gold = next(f for f in findings if "gold is Ag" in f["original_claim"])
    assert gold["external_sources"] == []  # all stripped
    assert gold["verification_method"] == "model_knowledge"
    assert "downgraded" in gold["limitations"]
