"""Background generation jobs for the chat page (follow-up task, section 7).

Why a worker thread: Streamlit runs one script per browser session and
processes widget clicks only BETWEEN script runs.  With the previous design
(the script blocking inside ``graph.invoke``) a Stop button click could
never be delivered - there was no way to cancel a running request.  Moving
generation into a background thread returns the script to the Streamlit
event loop immediately, so:

- the composer stays interactive and the Stop button works,
- a fragment (``ui.chat._generation_monitor``) polls the job and refreshes
  the page when it finishes,
- Stop sets the job's ``threading.Event``; graph nodes check it at safe
  points (see ``core/cancellation.py``) and wind the run down cleanly.

Thread-safety notes:
- The worker only touches thread-safe collaborators: the application
  ``Database`` (RLock-serialized), the LangGraph ``SqliteSaver`` checkpointer
  (its own internal lock serializes every operation), and Qdrant local mode.
- The worker NEVER calls Streamlit APIs (those are main-thread only); it
  reports completion through this module's registry.
- Jobs are keyed by a unique token stored in ``st.session_state``; the
  registry itself is guarded by a module lock.  Terminal jobs are pruned so
  a long-lived process does not accumulate them.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Job lifecycle statuses.
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

#: Terminal jobs older than this are pruned on the next registry access.
_PRUNE_AFTER_SECONDS = 600.0


@dataclass
class GenerationJob:
    """One generation request executing (or executed) in a worker thread."""

    token: str
    thread_id: str
    text: str
    started: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    status: str = RUNNING
    phase: str = "Thinking…"
    result: dict[str, Any] | None = None
    error: str | None = None
    finished: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -- worker-side updates (called from the worker thread) ------------------
    def report_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    @property
    def current_phase(self) -> str:
        with self.lock:
            return self.phase

    @property
    def terminal(self) -> bool:
        return self.status in (DONE, FAILED, CANCELLED)


_REGISTRY: dict[str, GenerationJob] = {}
_REGISTRY_LOCK = threading.Lock()


def start_generation(container, thread_id: str, text: str) -> str:
    """Spawn a worker that runs the workflow for ``text`` on ``thread_id``.

    Returns the unique job token; the UI keeps it in session state and the
    monitor fragment polls the job through it.
    """
    token = uuid.uuid4().hex
    job = GenerationJob(
        token=token, thread_id=thread_id, text=text, started=time.time()
    )
    with _REGISTRY_LOCK:
        _prune_old_locked()
        _REGISTRY[token] = job

    worker = threading.Thread(
        target=_run_job, args=(container, job), daemon=True,
        name=f"pkta-generation-{token[:8]}",
    )
    worker.start()
    return token


def get_job(token: str | None) -> GenerationJob | None:
    if not token:
        return None
    with _REGISTRY_LOCK:
        return _REGISTRY.get(token)


def dispose_job(token: str) -> None:
    """Remove a terminal job from the registry (after the UI consumed it)."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(token, None)


def request_stop(token: str) -> bool:
    """Signal cancellation to a running job.  Returns True if the job exists
    and had not been stopped yet."""
    job = get_job(token)
    if job is None or job.terminal:
        return False
    job.cancel_event.set()
    return True


def _prune_old_locked() -> None:
    """Drop terminal jobs older than the prune window (registry lock held)."""
    now = time.time()
    stale = [
        token
        for token, job in _REGISTRY.items()
        if job.terminal and job.finished is not None
        and now - job.finished > _PRUNE_AFTER_SECONDS
    ]
    for token in stale:
        _REGISTRY.pop(token, None)


def _run_job(container, job: GenerationJob) -> None:
    """Worker body: run the workflow, honouring the cancellation event.

    Uses ``send_message`` (the same entry point the synchronous flow used),
    so conversation bookkeeping, interrupts, and pending-operation recording
    behave exactly as before.  The only difference: it runs off the Streamlit
    script thread and honours Stop.
    """
    from core import cancellation
    from ui.graph_service import send_message

    cancellation.begin_generation(job.cancel_event, job.report_phase)
    try:
        result = send_message(container, job.thread_id, job.text)
        with job.lock:
            job.result = result
            job.status = (
                CANCELLED if job.cancel_event.is_set() else DONE
            )
            job.finished = time.time()
    except Exception as exc:  # noqa: BLE001 - surfaced by the UI monitor
        with job.lock:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = CANCELLED if job.cancel_event.is_set() else FAILED
            job.finished = time.time()
    finally:
        cancellation.end_generation()
