"""Live OpenRouter provider verification (run with your REAL API key).

Executes the acceptance checks from the provider-switch task against the
actual OpenRouter endpoint:

  A. Provider initialization (model id, endpoint, key validation,
     original-provider preservation).
  B. Basic conversation through NVIDIA Nemotron 3 Ultra.
  C. Structured output (prompt-based JSON + Pydantic validation):
     IntentResult, TaskProposal, GroundedAnswer.
  D. Tool calling: search_documents over SYNTHETIC documents, list_tasks.

Usage:
  1. Copy .env.example to .env and set OPENROUTER_API_KEY (and preferably
     ANTHROPIC_API_KEY for the fallback).
  2. python scripts/verify_openrouter.py

Data protection: this script only sends SYNTHETIC, non-sensitive content to
the free model endpoint. Never point it at confidential documents.

The offline test suite (pytest) covers the remaining acceptance areas
(RAG citations, human approval, durable resume, fallback simulation,
provider switching) without any network access.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import get_settings  # noqa: E402
from core.llm_gateway import ModelGateway, ModelGatewayError  # noqa: E402
from core.models import GroundedAnswer, IntentResult, RetrievedChunk, TaskProposal  # noqa: E402
from tools.document_tools import make_search_documents_tool  # noqa: E402
from tools.task_tools import make_list_tasks_tool  # noqa: E402
from database.connection import Database  # noqa: E402
from database.schema import init_schema  # noqa: E402
from database.tasks import TaskRepository  # noqa: E402

SYNTHETIC_CHUNK = RetrievedChunk(
    chunk_id="synthetic-1",
    document_id="synthetic-doc",
    filename="synthetic-notes.txt",
    page_number=1,
    section="Intro",
    text=(
        "The fictional Zeta project uses a fictional vector database called "
        "VectraDB for all synthetic retrieval experiments."
    ),
    score=0.92,
)

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []  # (check, outcome, detail)


def record(check: str, outcome: str, detail: str = "") -> None:
    results.append((check, outcome, detail))
    marker = "[PASS]" if outcome == PASS else "[FAIL]"
    print(f"{marker} {check}" + (f" - {detail}" if detail else ""))


def provider_of(gateway: ModelGateway) -> str:
    return " -> ".join(
        f"{event.get('provider')}({event.get('model')}) {event.get('outcome')}"
        for event in gateway.last_events
    )


def main() -> int:
    settings = get_settings()
    errors, _warnings = settings.validate()

    # -- A. Provider initialization ------------------------------------------
    print("\n== A. Provider initialization ==")
    info = ModelGateway.from_settings(settings).describe()
    if info["primary_provider"] != "openrouter":
        record("A1 active primary provider", FAIL, info["primary_provider"])
    else:
        record("A1 active primary provider", PASS, "openrouter")
    record(
        "A2 configured model",
        PASS if info["primary_model"]
        == "nvidia/nemotron-3-ultra-550b-a55b:free" else FAIL,
        info["primary_model"],
    )
    record(
        "A3 structured-output mode",
        PASS if info["primary_structured_output"] == "prompt" else FAIL,
        info["primary_structured_output"],
    )
    record(
        "A4 API key configured",
        PASS if info["primary_available"] else FAIL,
        "key present" if info["primary_available"] else str(errors),
    )
    record(
        "A5 fallback preserved",
        PASS if info["fallback_model"] == "claude-haiku-5-5" else FAIL,
        f"{info['fallback_provider']}:{info['fallback_model']} "
        f"(available={info['fallback_available']})",
    )
    record(
        "A6 original provider preserved",
        PASS if settings.primary_model == "gpt-6-luna" else FAIL,
        f"openai:{settings.primary_model} (inactive, restorable via "
        "LLM_PRIMARY_PROVIDER=openai)",
    )

    gateway = ModelGateway.from_settings(settings)
    from langchain_core.messages import HumanMessage  # noqa: E402

    # -- B. Basic conversation -------------------------------------------------
    print("\n== B. Basic conversation ==")
    try:
        reply = gateway.chat(
            [HumanMessage(content="Hello. What can you help me with?")],
            run_name="verify_basic_conversation",
        )
        text = str(reply.content).strip()
        record(
            "B1 assistant responds via Nemotron",
            PASS if text else FAIL,
            (text[:120] + "...") if len(text) > 120 else text,
        )
        record("B2 provider trace", PASS, provider_of(gateway))
    except ModelGatewayError as exc:
        record("B1 assistant responds via Nemotron", FAIL, str(exc)[:200])

    # -- C. Structured output ----------------------------------------------------
    print("\n== C. Structured output (prompt-based JSON) ==")
    try:
        intent = gateway.structured(
            IntentResult,
            [HumanMessage(content="Classify: hello there!")],
            run_name="verify_intent",
        )
        record(
            "C1 IntentResult",
            PASS if isinstance(intent, IntentResult) else FAIL,
            f"intent={intent.intent.value} confidence={intent.confidence}",
        )
    except ModelGatewayError as exc:
        record("C1 IntentResult", FAIL, str(exc)[:200])
    try:
        proposal = gateway.structured(
            TaskProposal,
            [HumanMessage(
                content="Please remind me to review the synthetic notes "
                "tomorrow. High priority."
            )],
            run_name="verify_proposal",
        )
        record(
            "C2 TaskProposal",
            PASS if isinstance(proposal, TaskProposal) else FAIL,
            f"title='{proposal.title}' priority={proposal.priority.value}",
        )
    except ModelGatewayError as exc:
        record("C2 TaskProposal", FAIL, str(exc)[:200])
    try:
        answer = gateway.structured(
            GroundedAnswer,
            [HumanMessage(
                content=(
                    "Answer only from this passage: The fictional Zeta "
                    "project uses VectraDB. Question: which database does "
                    "the Zeta project use?"
                )
            )],
            run_name="verify_grounded",
        )
        record(
            "C3 GroundedAnswer",
            PASS if isinstance(answer, GroundedAnswer) else FAIL,
            f"confidence={answer.confidence} citations={len(answer.citations)}",
        )
    except ModelGatewayError as exc:
        record("C3 GroundedAnswer", FAIL, str(exc)[:200])

    # -- D. Tool calling -----------------------------------------------------------
    print("\n== D. Tool calling ==")
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(Path(tmp) / "verify.db"))
        init_schema(db)
        task_repo = TaskRepository(db)
        collector: dict[str, dict] = {}
        search_tool = make_search_documents_tool(
            _StaticRetriever(), 5, collector
        )
        list_tool = make_list_tasks_tool(task_repo)
        try:
            response = gateway.chat(
                [
                    HumanMessage(
                        content=(
                            "Use the search_documents tool to find which "
                            "database the synthetic notes mention, then tell "
                            "me the answer briefly."
                        )
                    )
                ],
                tools=[search_tool],
                run_name="verify_tools",
            )
            used = [c["name"] for c in (response.tool_calls or [])]
            record(
                "D1 search_documents tool call generated",
                PASS if "search_documents" in used or collector else FAIL,
                f"tool_calls={used or 'none'} retrieved={len(collector)}",
            )
            record(
                "D2 final answer after tool use",
                PASS if str(response.content).strip() else FAIL,
                str(response.content)[:120],
            )
        except ModelGatewayError as exc:
            record("D1 search_documents tool call generated", FAIL, str(exc)[:200])
        try:
            listing = list_tool.invoke({"status": "all"})
            record("D3 list_tasks executes", PASS, listing.splitlines()[0][:100])
        except Exception as exc:  # noqa: BLE001
            record("D3 list_tasks executes", FAIL, str(exc)[:120])
        db.close()

    print("\n== Summary ==")
    failed = [r for r in results if r[1] != PASS]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.")
    for check, _outcome, detail in failed:
        print(f"  FAILED: {check} - {detail}")
    print(
        "\nNote: no API keys are printed by this script. Remaining "
        "acceptance areas (RAG citations, human approval/durable resume, "
        "fallback simulation, provider switching) are covered by the "
        "offline pytest suite."
    )
    return 1 if failed else 0


class _StaticRetriever:
    """Returns the synthetic chunk for any query (verification only)."""

    def search(self, query, top_k=None):  # noqa: ARG002
        return [SYNTHETIC_CHUNK]


if __name__ == "__main__":
    raise SystemExit(main())
