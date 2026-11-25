"""Deterministic claim/evidence comparison (follow-up task 4, sections 6.2, 8).

``verify_claims`` compares a factual claim against pieces of evidence
(document passages, model-knowledge statements, web excerpts) using a
DETERMINISTIC lexical analysis: shared key terms, numbers, named entities,
and directional phrases.  It never renders a final judgment itself - the
verdict stays with the model's structured reasoning - but it gives both
the model and the fact-check workflow an application-controlled
cross-check of what the evidence actually says.

The fact-check workflow also uses the same comparator core to VALIDATE
findings (e.g. that a finding's "document_evidence" really appears in the
cited document passage, and that cited external URLs were actually
retrieved - see graph.nodes / core.models FactCheckResult validation).
"""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Words that signal agreement/support vs. negation/contradiction.
_NEGATION_RE = re.compile(
    r"\b(not|no|never|neither|nor|isn't|is not|aren't|are not|wasn't|"
    r"was not|weren't|were not|cannot|can't|can not|don't|does not|"
    r"doesn't|didn't|did not|without|instead of|rather than|incorrect|"
    r"wrong|false|myth)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    """a an the and or but of in on at to for with by from as is are was
    were be been being it its this that these those his her their our your
    has have had do does did will would shall should can could may might
    not no nor so if then than there here when where which who whom whose
    what why how all any both each few more most other some such only own
    same too very s t just don now into about over under again further
    once during before after above below up down out off very""".split()
)


def _key_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\d+(?:\.\d+)?", text or "")
    }


class VerifyClaimsInput(BaseModel):
    claim: str = Field(
        min_length=5,
        max_length=1500,
        description="The factual claim to check, quoted verbatim.",
    )
    evidence: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Evidence texts to compare the claim against (document "
        "passages, reference statements, or web excerpts).",
    )


def compare_claim_to_evidence(
    claim: str, evidence: str
) -> dict:
    """Deterministic lexical comparison of one claim against one evidence.

    Returns ``{"overlap", "shared_terms", "shared_numbers",
    "negation_divergence", "signal"}`` where signal is one of
    ``consistent_signal`` / ``contradiction_signal`` / ``insufficient_overlap``.
    A *signal* is evidence-level only - never a verdict.
    """
    claim_terms = _key_terms(claim)
    evidence_terms = _key_terms(evidence)
    shared = claim_terms & evidence_terms
    overlap = (
        len(shared) / len(claim_terms) if claim_terms else 0.0
    )
    claim_numbers = _numbers(claim)
    evidence_numbers = _numbers(evidence)
    number_mismatch = bool(
        claim_numbers and evidence_numbers and claim_numbers != evidence_numbers
    )
    claim_negation = bool(_NEGATION_RE.search(claim))
    evidence_negation = bool(_NEGATION_RE.search(evidence))
    negation_divergence = claim_negation != evidence_negation

    if overlap < 0.2 and not shared:
        signal: str = "insufficient_overlap"
    elif number_mismatch or (negation_divergence and overlap >= 0.2):
        signal = "contradiction_signal"
    else:
        signal = "consistent_signal"
    return {
        "overlap": round(overlap, 3),
        "shared_terms": sorted(shared)[:12],
        "shared_numbers": sorted(claim_numbers & evidence_numbers),
        "number_mismatch": number_mismatch,
        "negation_divergence": negation_divergence,
        "signal": signal,
    }


def make_verify_claims_tool():
    @tool("verify_claims", args_schema=VerifyClaimsInput)
    def verify_claims(claim: str, evidence: list[str]) -> str:
        """Compare a factual claim against evidence texts, deterministically.

        Use during fact-checking to cross-check a document claim against
        reference knowledge or web excerpts.  Returns, per evidence item,
        the shared key terms, shared numbers, a negation/number-mismatch
        analysis, and an evidence-level signal (consistent_signal /
        contradiction_signal / insufficient_overlap).  These are signals,
        not verdicts: you must still reason about the final assessment.
        """
        comparisons = []
        for index, item in enumerate(evidence[:10], start=1):
            comparison = compare_claim_to_evidence(claim, item)
            comparison["evidence_index"] = index
            comparisons.append(comparison)
        return json.dumps(
            {
                "claim": claim,
                "note": "signals are lexical indicators, not verdicts",
                "comparisons": comparisons,
            },
            ensure_ascii=False,
        )

    return verify_claims
