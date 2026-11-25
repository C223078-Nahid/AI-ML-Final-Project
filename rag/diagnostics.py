"""Safe RAG diagnostics (follow-up task, sections 3.2.B and 3.2.E).

One logging facade for the document pipeline that reports, at INFO level:

- the Qdrant collection name and storage location actually in use,
- how many points the collection really contains (after ingestion),
- whether retrieval executed and how many chunks it returned,
- whether the retrieved context was included in the LLM request,
- which provider/model served the call.

Privacy rules:
- NO document content, chunk text, or query text is ever logged (only
  lengths and counts).
- NO credentials or configuration secrets are ever logged.
- Infrastructure failures are logged distinctly from "no matching
  content" so the two can never be confused again.

Enable in the terminal with:  PKTA_RAG_LOG=INFO  (or set the standard
``logging`` level for the ``pkta.rag`` logger).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.models import RetrievedChunk

logger = logging.getLogger("pkta.rag")


def _level_from_env() -> int | None:
    """Optional PKTA_RAG_LOG override (DEBUG/INFO/WARNING/...)."""
    raw = (os.environ.get("PKTA_RAG_LOG") or "").strip().upper()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    return getattr(logging, raw, None)


def attach_stderr_handler() -> None:
    """Make the diagnostics visible in the ``streamlit run`` terminal.

    Idempotent (safe across Streamlit reruns): attaches one stderr handler
    to the ``pkta.rag`` logger only. Other libraries' loggers are untouched.
    The level is INFO by default and can be overridden with PKTA_RAG_LOG.
    """
    if getattr(logger, "_pkta_stderr_attached", False):
        return
    level = _level_from_env()
    logger.setLevel(level if level is not None else logging.INFO)
    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger._pkta_stderr_attached = True  # type: ignore[attr-defined]


# Library default: never raise, never print unless explicitly configured.
logger.addHandler(logging.NullHandler())


def log_library_status(store: Any, document_count: int) -> None:
    """Report the actual shared-collection state (task section 3.2.B)."""
    try:
        stats = store.stats() if hasattr(store, "stats") else {}
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        logger.warning("[rag-diag] could not read vector store stats: %s", exc)
        return
    logger.info(
        "[rag-diag] Qdrant collection name: %s | storage location: %s | "
        "indexed points: %s | documents in library: %s | embedding dim: %s",
        stats.get("collection"),
        stats.get("path"),
        stats.get("points"),
        document_count,
        stats.get("dimension"),
    )


def log_ingestion(store: Any, filename: str, document_id: str, chunks: int) -> None:
    """Report one completed ingestion with verification counts."""
    try:
        stats = store.stats() if hasattr(store, "stats") else {}
        points = stats.get("points")
    except Exception as exc:  # noqa: BLE001
        points = f"unavailable ({exc})"
    logger.info(
        "[rag-diag] ingestion finished: file=%s document_id=%s chunks=%s | "
        "collection points now: %s",
        filename,
        document_id,
        chunks,
        points,
    )


def log_retrieval(query_source: str, chunks: list["RetrievedChunk"]) -> None:
    """Report a retrieval execution (task section 3.2.E)."""
    names = sorted({c.filename for c in chunks})
    logger.info(
        "[rag-diag] Document retrieval executed: Yes | query source: %s | "
        "retrieved chunk count: %d | retrieved document names: %s",
        query_source,
        len(chunks),
        names,
    )


def log_retrieval_failure(stage: str, error: Exception) -> None:
    """Distinguish infrastructure failures from ordinary missing evidence."""
    logger.error(
        "[rag-diag] retrieval infrastructure failure at '%s': %s: %s",
        stage,
        type(error).__name__,
        str(error)[:200],
    )


def log_context_construction(
    chunks: list["RetrievedChunk"], provider: str, model: str
) -> None:
    """Confirm the retrieved context actually entered the LLM request."""
    total_chars = sum(len(c.text) for c in chunks)
    estimated_tokens = max(1, (total_chars + 3) // 4)
    logger.info(
        "[rag-diag] Context included in LLM request: Yes | passages: %d | "
        "context length: %d chars (~%d tokens) | selected model provider: %s | "
        "selected model: %s",
        len(chunks),
        total_chars,
        estimated_tokens,
        provider,
        model,
    )


def log_model_invocation(run_name: str, provider: str, model: str, ok: bool,
                         error: str | None = None) -> None:
    """Report the outcome of one gateway call from the RAG path."""
    if ok:
        logger.info(
            "[rag-diag] Model invocation completed: Yes | run: %s | "
            "provider: %s | model: %s",
            run_name,
            provider,
            model,
        )
    else:
        logger.error(
            "[rag-diag] Model invocation failed | run: %s | provider: %s | "
            "model: %s | error: %s",
            run_name,
            provider,
            model,
            (error or "")[:200],
        )


def log_citation_validation(
    problem: str | None,
    valid: int,
    dropped: int,
    recovery: str | None,
) -> None:
    """Report whether the model's citations resolved to real passages."""
    if problem is None:
        logger.info(
            "[rag-diag] Evidence validation passed: Yes | valid citations: %d"
            "%s",
            valid,
            f" | corrected via {recovery}" if recovery else "",
        )
    else:
        logger.warning(
            "[rag-diag] Evidence validation problem: %s | valid citations: %d"
            " | dropped: %d",
            problem,
            valid,
            dropped,
        )


def log_final_outcome(kind: str, insufficient_evidence: bool,
                      citations: int, confidence: float | None) -> None:
    """One summary line per grounded answer: the decision the user saw."""
    logger.info(
        "[rag-diag] Final answer generated successfully: %s | kind: %s | "
        "insufficient_evidence: %s | citations: %d | confidence: %s",
        "Yes" if not insufficient_evidence else "No (abstention)",
        kind,
        insufficient_evidence,
        citations,
        f"{confidence:.2f}" if confidence is not None else "n/a",
    )


def log_reconciliation(report: dict) -> None:
    """Report the startup DB<->Qdrant consistency check."""
    if report.get("orphans_removed"):
        logger.warning(
            "[rag-diag] store reconciliation removed %s orphan point group(s) "
            "(scanned %s points; %s points remain)",
            report.get("orphans_removed"),
            report.get("scanned"),
            report.get("points"),
        )
    else:
        logger.info(
            "[rag-diag] store reconciliation clean | scanned: %s points | "
            "points: %s",
            report.get("scanned"),
            report.get("points"),
        )


def log_routing(intent: str, source: str) -> None:
    """Report the final routing decision and where it came from."""
    logger.info(
        "[rag-diag] routing decision: %s | decided by: %s", intent, source
    )


def log_factcheck_validation(
    claims: int, stripped_sources: int, notes: list[str] | None = None
) -> None:
    """Fact-check evidence-validation outcome (follow-up task 4, 13).

    Safe metadata only: counts and note summaries - never document
    passages or claim text.
    """
    logger.info(
        "factcheck validation: claims=%d stripped_sources=%d notes=%s",
        claims, stripped_sources, "; ".join((notes or [])[:3]),
    )
