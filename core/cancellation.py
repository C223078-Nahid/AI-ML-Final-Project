"""Cooperative cancellation for LangGraph generation runs (follow-up task,
section 7).

Why cooperative: Streamlit runs one script per browser session and cannot
process a widget click while that script is blocked inside
``graph.invoke(...)``.  The chat execution model therefore moved generation
into a background worker thread (``ui/generation.py``) so the script returns
to the Streamlit event loop and the Stop button stays clickable.  The worker
thread - and the graph nodes executing inside it - check a per-run
``threading.Event`` at well-defined points:

- at the entry of every workflow node,
- before every model call in the bounded tool loops,
- before every tool invocation in those loops,
- immediately before the irreversible ``create_task`` write.

The one thing this design cannot do is abort a single in-flight HTTPS
request: the synchronous OpenAI/Anthropic clients used by the installed
``langchain-openai`` / ``langchain-anthropic`` versions expose no cancellation
handle for a blocking call (their async counterparts would, but the whole
application runs on the synchronous LangGraph invoke path).  When Stop is
clicked mid-request, the current model call finishes (bounded by
``MODEL_TIMEOUT_SECONDS``), its output is discarded, every subsequent node
short-circuits to ``final_response``, which emits a "stopped" marker message.
No tool executes afterwards, no task is written, the checkpointer stays
consistent (the graph completes its normal path instead of being killed
mid-superstep), and no stale generation flags remain.

The same module carries an optional progress-phase callback: nodes report
what the application is actually doing ("Searching uploaded documents...",
"Preparing response...") and the UI displays it under the thinking dots.
These are real application events - never fabricated model reasoning.

Thread-safety: the state lives in a ``threading.local``, so the worker
thread's flag can never leak into the main script thread (which renders
pages and serves approval clicks) or into other sessions' workers.
"""

from __future__ import annotations

import threading
from typing import Callable

_local = threading.local()


def begin_generation(
    cancel_event: threading.Event,
    phase_callback: Callable[[str], None] | None = None,
) -> None:
    """Register the cancellation event (and optional phase reporter) that
    graph nodes running in THIS thread should honour."""
    _local.cancel_event = cancel_event
    _local.phase_callback = phase_callback


def end_generation() -> None:
    """Clear the per-thread generation context (worker thread shutdown)."""
    _local.cancel_event = None
    _local.phase_callback = None


def cancellation_requested() -> bool:
    """True when the user requested cancellation for the running generation.

    Always False on threads without an active generation (the main script
    thread, approval-resume runs, tests) - those code paths are unchanged.
    """
    event = getattr(_local, "cancel_event", None)
    return event is not None and event.is_set()


def report_phase(phase: str) -> None:
    """Report an actual application progress phase, if a listener exists."""
    callback = getattr(_local, "phase_callback", None)
    if callback is not None:
        try:
            callback(phase)
        except Exception:  # noqa: BLE001 - progress reporting must never fail a run
            pass
