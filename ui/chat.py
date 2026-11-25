"""Chat workspace: the primary application screen (ChatGPT-inspired).

Layout (follow-up task, sections 4-5 and 7-8):
- conversation transcript in the main area,
- attachment previews ABOVE the message input (chips with file-type icon,
  indexing status, and a remove affordance),
- a composer row: "+" attachment control on the left (a collapsed
  ``st.file_uploader`` styled as a compact square - ONE click opens the
  file browser, no popover, no caret), the text input in the middle, and
  Send/Stop on the right,
- drag-and-drop upload across the whole chat/message area (custom
  same-origin component, ``ui/dropzone.py``) with a clear drag-over
  overlay,
- a visible progress state (thinking dots + the ACTUAL application phase:
  "Searching uploaded documents...", "Retrieving relevant passages...",
  "Preparing response...") while the assistant works.

New-chat screen (follow-up task 4, sections 4-5):
- the welcome heading, subtitle, and the suggested-prompt buttons are
  horizontally CENTERED in the chat area,
- every suggested prompt is CLICKABLE: clicking inserts its text into the
  composer draft (the shared ``pkta_composer_input`` key), ready to edit
  and send - it never auto-submits and never creates a chat.  An existing
  draft is preserved: the suggestion is appended on a new line instead of
  overwriting what the user typed.

Attachments (follow-up task, sections 4-5): files picked through "+" or
dropped on the chat area are processed through the EXISTING shared upload
pipeline IMMEDIATELY (validate -> parse -> chunk -> embed -> index -> live
staged progress).  The chip above the composer is a reference into the
globally shared library - every chat session can search the document
through RAG; removing a chip only unstages the preview, the document stays
in the library (manage deletion on the Documents page).

Request cancellation (follow-up task, section 7): generation runs in a
background worker thread (``ui/generation.py``), so the script returns to
the Streamlit event loop and the composer stays interactive:
- while generating, Send is REPLACED by a Stop button and the input shows
  a "Generating response..." placeholder,
- Stop signals the cooperative cancellation event; graph nodes stop at the
  next safe point (no further model calls, tools, or task writes), and the
  assistant message becomes an explicit stopped marker,
- after Stop (or completion) the composer is re-enabled; a message
  submitted while a stopped run is still winding down is queued and sent
  automatically as soon as the worker exits,
- the ``@st.fragment`` monitor below polls the job, shows live progress,
  and triggers a full rerun when the response is ready.

Conversation memory is per-thread (LangGraph checkpoints); the document
library is shared across all threads, so document questions work in every
chat without re-uploading anything.
"""

from __future__ import annotations

import html
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from ui import generation
from ui.approvals import render_proposal_card
from ui.graph_service import (
    get_thread_messages,
    new_thread_id,
    pending_interrupt,
)
from ui.uploads import process_upload

#: Session-state key of the composer text input.  It is the ONE shared
#: draft state: the suggested-prompt buttons write into it, and the
#: composer's st.text_input renders from it (follow-up task, section 4 -
#: "the chat input and suggested prompts must use the same shared state so
#: that inserted text is immediately visible").
COMPOSER_INPUT_KEY = "pkta_composer_input"

#: Session-state flag set by the form's on_click callback (which runs
#: BEFORE the script rerun, where widget keys may still be written) and
#: consumed by the main body right after the form to submit the message.
COMPOSER_PENDING_SEND = "pkta_composer_pending_send"

MAX_DISPLAY_CHARS = 12000

SUGGESTIONS = [
    "Summarize my uploaded document",
    "What are the main topics in my notes?",
    "Which tasks are currently open?",
    "What day is today?",
]

SUPPORTED_TYPES = ["pdf", "docx", "md", "txt"]

#: A double submission (Enter pressed twice, or Send clicked twice) within
#: this window for the same thread+text is dropped as a duplicate.
DEDUP_WINDOW_SECONDS = 4.0

#: File-type glyphs for attachment chips (consistent with the sidebar icons).
_FILE_ICONS = {".pdf": "PDF", ".docx": "DOC", ".txt": "TXT", ".md": "MD"}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _escape(text: str) -> str:
    return html.escape(str(text or ""))


def _render_user_message(message: HumanMessage) -> None:
    text = str(message.content or "")[:MAX_DISPLAY_CHARS]
    st.markdown(
        f'<div class="pkta-user-row"><div class="pkta-user-bubble">'
        f"{_escape(text)}</div></div>",
        unsafe_allow_html=True,
    )


def _citation_card(index: int, citation: dict) -> str:
    filename = _escape(citation.get("filename", "unknown"))
    location = ""
    if citation.get("page_number") is not None:
        location = f" — Page {_escape(str(citation['page_number']))}"
    elif citation.get("section"):
        location = f" — {_escape(str(citation['section']))}"
    passage = _escape(str(citation.get("supporting_passage") or "")[:600])
    meta_parts = []
    if citation.get("chunk_id"):
        meta_parts.append(f"chunk {_escape(str(citation['chunk_id']))}")
    if citation.get("score") is not None:
        meta_parts.append(f"score {float(citation['score']):.2f}")
    meta = f" · {_escape(' · '.join(meta_parts))}" if meta_parts else ""
    return (
        f'<div class="pkta-source-item">'
        f'<div class="pkta-source-head">{index}. {filename}{location}</div>'
        f'<div class="pkta-passage">{passage}</div>'
        f'<div class="pkta-meta">{meta}</div>'
        f"</div>"
    )


def _render_citations(payload: dict) -> None:
    citations = payload.get("citations") or []
    if not citations:
        return
    cards = "".join(
        _citation_card(index, citation)
        for index, citation in enumerate(citations, start=1)
    )
    st.markdown(
        f'<div class="pkta-sources"><div style="font-size:0.8rem;'
        f'color:#9b9b9b;margin-bottom:0.1rem;">Sources</div>'
        f"{cards}</div>",
        unsafe_allow_html=True,
    )


def _render_tool_results(payload: dict) -> None:
    tool_results = payload.get("tool_results") or []
    if not tool_results:
        return
    chips = "".join(
        f'<span class="pkta-tool-chip">'
        f"{_escape(str(trace.get('tool')))}"
        f"({_escape(str(trace.get('input') or ''))[:60]})</span>"
        for trace in tool_results
    )
    st.markdown(
        f'<div class="pkta-tool-trace">{chips}</div>', unsafe_allow_html=True
    )


def _render_tasks(payload: dict) -> None:
    tasks = payload.get("tasks") or []
    if not tasks:
        return
    rows = [
        {
            "Title": task.get("title"),
            "Priority": task.get("priority"),
            "Status": task.get("status"),
            "Due": task.get("due_date") or "—",
        }
        for task in tasks
    ]
    st.table(rows)


# ---------------------------------------------------------------------------
# Fact-check report (follow-up task 4, section 10)
# ---------------------------------------------------------------------------
#: Restrained status labels (read-only report; original documents are
#: never modified by fact-checking).
_STATUS_LABELS = {
    "correct": ("CORRECT", "pkta-fc-correct"),
    "incorrect": ("INCORRECT", "pkta-fc-incorrect"),
    "partially_correct": ("PARTIALLY CORRECT", "pkta-fc-partial"),
    "misleading": ("MISLEADING", "pkta-fc-partial"),
    "outdated": ("OUTDATED", "pkta-fc-partial"),
    "contradictory": ("CONTRADICTORY", "pkta-fc-incorrect"),
    "unverified": ("UNVERIFIED", "pkta-fc-unverified"),
    "opinion": ("OPINION", "pkta-fc-unverified"),
    "ambiguous": ("AMBIGUOUS", "pkta-fc-unverified"),
}

_VERIFICATION_LABELS = {
    "model_knowledge": ("Model knowledge", "Assessed against the model's "
                        "general knowledge - no independent web source."),
    "model_knowledge_and_external": (
        "Model knowledge + independent sources",
        "Assessed against general knowledge AND retrieved external "
        "evidence.",
    ),
    "external_only": ("Independent sources", "Decided from retrieved "
                      "external evidence."),
    "document_only": ("Document evidence", "About the document itself "
                      "(what/where it says something)."),
}


def _render_fact_check_report(payload: dict) -> None:
    """Cross-reference report: what the document says vs what is supported.

    Read-only presentation with three clearly separated evidence layers:
    document evidence, model knowledge, and independent external sources.
    """
    report = payload.get("report") or {}
    findings = report.get("findings") or []
    if not findings:
        return

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.get("status", "unverified")] = (
            counts.get(finding.get("status", "unverified"), 0) + 1
        )
    summary = " · ".join(
        f"{count} {status.replace('_', ' ')}"
        for status, count in sorted(counts.items())
    )
    st.markdown(
        f'<div class="pkta-fc-head">FACT-CHECK REPORT — {len(findings)} '
        f"claim(s): {_escape(summary)}</div>",
        unsafe_allow_html=True,
    )

    for index, finding in enumerate(findings, start=1):
        status_key = finding.get("status") or "unverified"
        label, css = _STATUS_LABELS.get(status_key, status_key.upper())
        method = str(finding.get("verification_method") or "model_knowledge")
        method_label, method_hint = _VERIFICATION_LABELS.get(
            method, (method, "")
        )
        source = finding.get("filename") or "unknown document"
        location = ""
        if finding.get("page_number") is not None:
            location = f" — Page {finding['page_number']}"
        elif finding.get("section"):
            location = f" — {finding['section']}"
        confidence = finding.get("confidence")
        confidence_txt = (
            f" · confidence {float(confidence):.0%}" if confidence is not None else ""
        )

        with st.expander(
            f"Claim {index}: {label} — “"
            f"{_escape(str(finding.get('original_claim', ''))[:110])}”",
            expanded=(index == 1),
        ):
            st.markdown(
                f'<span class="pkta-fc-status {css}">{label}</span>'
                f'<span class="pkta-fc-method" title="{_escape(method_hint)}">'
                f"{_escape(method_label)}</span>",
                unsafe_allow_html=True,
            )
            claim_text = str(finding.get("original_claim", ""))
            st.markdown(f"**Document claim:** {_escape(claim_text)}")
            st.caption(
                f"Source: {_escape(source)}{_escape(location)}"
                f"{_escape(confidence_txt)}"
            )
            corrected = finding.get("corrected_statement")
            if corrected:
                st.markdown(
                    f"**Correct information:** {_escape(str(corrected))}"
                )
            explanation = finding.get("explanation")
            if explanation:
                st.markdown(f"**Explanation:** {_escape(str(explanation))}")
            evidence = finding.get("document_evidence")
            if evidence:
                st.markdown(
                    '<div class="pkta-fc-evidence">'
                    f"<b>Document evidence:</b> {_escape(str(evidence))}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            external = finding.get("external_sources") or []
            if external:
                st.markdown("**Independent sources (actually retrieved):**")
                for source_item in external:
                    title = _escape(str(source_item.get("title", "")))
                    url = _escape(str(source_item.get("url", "")))
                    domain = _escape(str(source_item.get("domain", "")))
                    excerpt = _escape(
                        str(source_item.get("excerpt", ""))[:400]
                    )
                    published = source_item.get("publication_date")
                    retrieved = source_item.get("retrieval_date")
                    dating = " · ".join(
                        part
                        for part in (
                            f"published {published}" if published else "",
                            f"retrieved {retrieved}" if retrieved else "",
                        )
                        if part
                    )
                    st.markdown(
                        f'<div class="pkta-fc-source">'
                        f'<a href="{url}" target="_blank" rel="noreferrer">'
                        f"{title}</a>"
                        + (f" — {_escape(domain)}" if domain else "")
                        + (f" · {dating}" if dating else "")
                        + (f'<div class="pkta-fc-excerpt">“{excerpt}”</div>'
                           if excerpt else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(
                    "No independent external source was retrieved for this "
                    "finding — the assessment rests on document evidence "
                    "and the model's general knowledge."
                )
            limitations = finding.get("limitations")
            if limitations:
                st.caption(f"Limitations: {_escape(str(limitations))}")

    for note in report.get("search_notes") or []:
        st.caption(_escape(str(note)))


def _render_ai_payload(payload: dict | None) -> None:
    if not payload:
        return
    kind = payload.get("kind")
    if kind == "answer":
        if payload.get("insufficient_evidence"):
            if payload.get("library_empty"):
                st.info(
                    "No documents are indexed yet. Attach one with the + "
                    "button or drop it anywhere in the chat — it becomes "
                    "searchable in every conversation."
                )
            else:
                st.info(
                    "The indexed documents do not contain sufficient "
                    "support for this answer."
                )
        elif payload.get("unverified"):
            st.caption(
                "Answer kept, but its citation could not be verified "
                "against the retrieved passages."
            )
        _render_citations(payload)
    elif kind in ("task_created", "task_list"):
        _render_tasks(payload)
    elif kind == "fact_check_report":
        _render_fact_check_report(payload)
    elif kind == "stopped":
        st.markdown(
            '<div class="pkta-stopped-marker">■ Response stopped</div>',
            unsafe_allow_html=True,
        )
    _render_tool_results(payload)


def render_message(message) -> None:
    if isinstance(message, HumanMessage):
        _render_user_message(message)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            text = str(message.content or "")
            if text:
                st.write(text[:MAX_DISPLAY_CHARS])
            _render_ai_payload(message.additional_kwargs.get("pkta_payload"))


def _render_thinking(label: str = "Thinking…") -> None:
    """The visible working indicator shown while a request is executing.

    ``label`` is the ACTUAL current application phase, reported by the
    graph nodes through the generation job (never a fabricated reasoning
    process - see follow-up task, section 8).
    """
    st.markdown(
        '<div class="pkta-thinking" aria-live="polite">'
        '<span class="pkta-thinking-dots">'
        '<span class="pkta-dot"></span><span class="pkta-dot"></span>'
        '<span class="pkta-dot"></span></span>'
        f"<span>{_escape(label)}</span></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Empty state (new chat)
# ---------------------------------------------------------------------------
def apply_suggestion(text: str) -> None:
    """Insert a suggested prompt into the composer (follow-up task, 4).

    Clicking a suggestion only fills the input - it never submits and never
    creates a new chat.  Sensible draft behaviour: an empty composer is
    filled with the prompt; a non-empty draft is PRESERVED and the
    suggestion is appended on a new line (nothing the user typed is
    overwritten); an identical draft is left untouched.
    """
    current = (st.session_state.get(COMPOSER_INPUT_KEY) or "").strip()
    if not current:
        st.session_state[COMPOSER_INPUT_KEY] = text
        return
    # Already present (or it IS the whole draft): no-op.
    if text in current or current == text:
        return
    # Non-empty draft: PRESERVE it and append the suggestion (the composer
    # is a single-line input, so the parts are joined with a space -
    # nothing the user typed is ever overwritten).
    st.session_state[COMPOSER_INPUT_KEY] = f"{current} {text}"


def _render_empty_state() -> None:
    st.markdown(
        '<div class="pkta-empty">'
        '<div class="pkta-hello">What can I help with?</div>'
        '<div class="pkta-subtitle">'
        "Ask about your documents, manage tasks, or just chat. "
        "Use the + button or drag a document anywhere to make it "
        "searchable in every conversation."
        "</div></div>",
        unsafe_allow_html=True,
    )
    # Clickable suggested prompts (follow-up task, section 4): each button
    # inserts its text into the composer draft (editable, NOT auto-sent).
    # The grid sits in a centered guttered column so the suggestion block
    # is horizontally centered in the chat area and stays responsive.
    _gutter_l, core, _gutter_r = st.columns([0.9, 4.6, 0.9])
    with core:
        row_a = st.columns(2)
        row_b = st.columns(2)
        placements = (
            (row_a[0], SUGGESTIONS[0]),
            (row_a[1], SUGGESTIONS[1]),
            (row_b[0], SUGGESTIONS[2]),
            (row_b[1], SUGGESTIONS[3]),
        )
        for index, (column, prompt) in enumerate(placements):
            with column:
                st.button(
                    prompt,
                    key=f"pkta_sugg_{index}",
                    on_click=apply_suggestion,
                    args=(prompt,),
                    help="Insert this prompt into the message box - you can "
                    "edit it before sending.",
                    use_container_width=True,
                )
    st.markdown(
        "<div style='height:0.9rem;'></div>", unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Attachments: immediate ingestion + preview chips above the composer
# ---------------------------------------------------------------------------
def _file_icon(name: str) -> str:
    suffix = "." + str(name).rsplit(".", 1)[-1].lower() if "." in name else ""
    return _FILE_ICONS.get(suffix, "FILE")


def _stage_attachment(name: str, size: int, status: str,
                      document_id: str) -> None:
    """Record an ingested document as a composer attachment chip.

    The chip is a REFERENCE into the shared library (the document is
    already indexed by the time a chip exists).  Removing a chip only
    unstages the preview - the document stays in the library until deleted
    on the Documents page.
    """
    staged = st.session_state.setdefault("pkta_pending_attachments", [])
    if any(a["name"] == name and a["size"] == size for a in staged):
        return
    staged.append(
        {"name": name, "size": size, "status": status, "document_id": document_id}
    )


def _ingest_and_stage(container, name: str, data: bytes) -> None:
    """Run one file through the EXISTING upload pipeline, then stage a chip.

    ``process_upload`` is the single ingestion entry point shared with the
    Documents page (live staged progress, duplicate detection, rerun
    safety).  A chip is only staged when the file really is in the library
    (indexed, or a duplicate that was already indexed earlier).
    """
    result = process_upload(container, name, data)
    if result.status in ("indexed", "duplicate"):
        _stage_attachment(
            result.filename or name,
            len(data),
            result.status,
            result.document_id,
        )


def _handle_attachment_selection(container, files) -> None:
    """Files picked through the + button: ingest immediately, then reset
    the uploader widget so the same file can be picked again later."""
    for uploaded in files:
        _ingest_and_stage(
            container, uploaded.name, uploaded.getvalue()
        )
    st.session_state["pkta_uploader_seq"] = (
        st.session_state.get("pkta_uploader_seq", 0) + 1
    )
    st.rerun()


def _handle_dropped_files(container, pairs) -> None:
    """Files dropped anywhere on the chat area (component callback)."""
    for name, data in pairs:
        _ingest_and_stage(container, name, data)
    st.rerun()


def _render_attachment_chips(processing: bool) -> None:
    """Attachment previews above the composer.

    Each chip shows the file-type badge, the filename, and its indexing
    status; clicking the chip removes it from the preview (the document
    itself stays in the shared library).
    """
    staged = st.session_state.get("pkta_pending_attachments") or []
    if not staged:
        return
    cols = st.columns(min(len(staged), 4))
    removed = False
    for i, col in enumerate(cols):
        with col:
            att = staged[i]
            badge = _file_icon(att["name"])
            status = (
                "indexed ✓"
                if att.get("status") == "indexed"
                else "already indexed"
            )
            label = f"[{badge}]  {att['name']}  ·  {status}  ✕"
            if st.button(
                label,
                key=f"pkta_attach_remove_{i}_{att['size']}",
                help=(
                    f"Remove {att['name']} from this message. The document "
                    "remains in the shared library (delete it on the "
                    "Documents page if you want it gone everywhere)."
                ),
                disabled=processing,
                use_container_width=True,
            ):
                staged.pop(i)
                removed = True
    if removed:
        st.rerun()


# ---------------------------------------------------------------------------
# Generation jobs (background worker + monitor fragment)
# ---------------------------------------------------------------------------
def _consume_terminal_job(token: str, job) -> None:
    """Move a finished job's outcome into session state and clean up."""
    st.session_state["pkta_processing"] = None
    st.session_state["pkta_last_outcome"] = {
        "status": job.status,
        "error": job.error,
        "thread_id": job.thread_id,
    }
    generation.dispose_job(token)


@st.fragment(run_every=0.5)
def _generation_monitor(token: str) -> None:
    """Live progress for the running generation (reruns twice a second).

    Shows the thinking indicator with the REAL application phase, and when
    the job reaches a terminal state, consumes it and triggers a full app
    rerun so the transcript, sidebar, and composer all refresh.
    """
    job = generation.get_job(token)
    if job is None:
        return
    if not job.terminal:
        label = (
            "Stopping…"
            if job.cancel_event.is_set()
            else job.current_phase
        )
        with st.chat_message("assistant"):
            _render_thinking(label)
        return
    # Terminal: consume and refresh the whole page (the checkpoint now
    # holds the final assistant message - or the stopped marker).
    _consume_terminal_job(token, job)
    st.rerun(scope="app")


# ---------------------------------------------------------------------------
# Composer (attachments above, + left, input, Send/Stop right)
# ---------------------------------------------------------------------------
def _active_job_for(thread_id: str | None):
    """The running/stopping job for THIS thread, if any."""
    token = st.session_state.get("pkta_processing")
    if not token:
        return None, None
    job = generation.get_job(token)
    if job is None or job.thread_id != thread_id:
        return None, token
    return job, token


def _submit_message(container, thread_id: str | None, text: str) -> None:
    """Register the submitted message and start (or queue) generation."""
    # Dedup: an identical request in the same thread within the window is a
    # double submission, not a new message.
    last = st.session_state.get("pkta_last_request")
    if (
        last
        and last[0] == (thread_id or "")
        and last[1] == text
        and time.time() - last[2] < DEDUP_WINDOW_SECONDS
    ):
        return
    st.session_state["pkta_last_request"] = (
        thread_id or "",
        text,
        time.time(),
    )
    if not thread_id:
        thread_id = new_thread_id()
        st.session_state["pkta_thread"] = thread_id

    # Attachment chips were already ingested at selection time; they belong
    # to this message and clear now.
    st.session_state["pkta_pending_attachments"] = []

    # If a (stopped) generation is still winding down on this thread, queue
    # the message; the monitor fragment starts it as soon as the worker
    # exits, so nothing is lost and no two runs share a thread.
    job, _token = _active_job_for(thread_id)
    if job is not None and not job.terminal:
        st.session_state["pkta_queued"] = {
            "thread_id": thread_id,
            "text": text,
        }
        st.rerun()

    token = generation.start_generation(container, thread_id, text)
    st.session_state["pkta_processing"] = token
    st.rerun()


def _maybe_start_queued(container, thread_id: str | None) -> None:
    """Start a queued message once its thread has no active generation."""
    queued = st.session_state.get("pkta_queued")
    if not queued or queued.get("thread_id") != thread_id:
        return
    job, _token = _active_job_for(thread_id)
    if job is not None and not job.terminal:
        return  # still winding down; the monitor will re-check via rerun
    st.session_state["pkta_queued"] = None
    token = generation.start_generation(
        container, queued["thread_id"], queued["text"]
    )
    st.session_state["pkta_processing"] = token
    st.rerun()


def _on_composer_submit(container, thread_id: str | None) -> None:
    """on_click callback of the Send button (runs BEFORE the rerun).

    Callbacks are the one place where a widget key may be written after the
    previous run: the draft is read from the shared composer key, staged
    for submission, and the visible box is cleared - so the input both
    submits and empties in a single interaction.
    """
    draft = (st.session_state.get(COMPOSER_INPUT_KEY) or "").strip()
    if draft:
        st.session_state[COMPOSER_PENDING_SEND] = draft
        st.session_state[COMPOSER_INPUT_KEY] = ""


def _render_composer(container, thread_id: str | None, job, token: str | None,
                     approval_pending: bool) -> None:
    """The composer: attachment previews above, + / input / Send-Stop row.

    Three states:
    - idle: enabled input + Send,
    - generating: disabled input ("Generating response...") + Stop in the
      Send position,
    - stopping (Stop clicked, worker winding down): enabled input + Send;
      submissions are queued until the worker exits.

    The text input is KEYED (``pkta_composer_input``): that key is the
    shared draft state the suggested-prompt buttons write into, so an
    inserted prompt is immediately visible in the box (follow-up task,
    section 4).  Submission runs through the form's on_click callback,
    which reads the draft, stages it, and clears the box.
    """
    generating = job is not None and not job.terminal
    stopping = generating and job.cancel_event.is_set()
    disabled = (generating and not stopping) or approval_pending

    _render_attachment_chips(disabled)

    with st.container():
        st.markdown(
            '<div class="pkta-composer-bar"></div>', unsafe_allow_html=True
        )
        # The plus column needs enough width for the 2.35rem square so it
        # never clips behind the input (0.32 ratio ~ 45px at typical
        # widths; narrower windows are handled by the media query).
        col_plus, col_form = st.columns([0.32, 5.2], gap="small",
                                        vertical_alignment="bottom")
        with col_plus:
            # ONE click opens the file browser directly: a collapsed-label
            # st.file_uploader whose dropzone is styled as the compact "+"
            # square (no popover, no extra caret).  The widget key is
            # bumped after every selection so the same file can be picked
            # again later.
            uploaded = st.file_uploader(
                "Attach documents",
                type=SUPPORTED_TYPES,
                accept_multiple_files=True,
                key=(
                    "pkta_composer_uploader"
                    f"{st.session_state.get('pkta_uploader_seq', 0)}"
                ),
                label_visibility="collapsed",
                disabled=disabled,
                help="Attach documents (PDF, DOCX, Markdown, TXT) - "
                "shared across every chat",
            )
            if uploaded:
                _handle_attachment_selection(container, uploaded)
        with col_form:
            if approval_pending:
                st.text_input(
                    "Message",
                    value="",
                    disabled=True,
                    placeholder="Approve or reject the pending task "
                    "proposal first.",
                    label_visibility="collapsed",
                )
                return
            if generating and not stopping:
                # GENERATING: input disabled, Stop replaces Send.
                col_text, col_stop = st.columns(
                    [5.4, 0.78], gap="small", vertical_alignment="bottom"
                )
                with col_text:
                    st.text_input(
                        "Message",
                        value="",
                        disabled=True,
                        placeholder="Generating response...",
                        label_visibility="collapsed",
                    )
                with col_stop:
                    if st.button(
                        "Stop",
                        key="pkta_composer_stop",
                        icon=":material/stop:",
                        help="Stop generating the response",
                        use_container_width=True,
                    ):
                        if token and generation.request_stop(token):
                            pass  # signal sent; monitor finishes the rest
                        st.rerun()
                return
            if stopping:
                st.caption(
                    "Stopping the previous response — your next message "
                    "will be sent as soon as it exits."
                )
            with st.form("pkta_composer", border=False):
                col_text, col_send = st.columns(
                    [5.4, 0.78], gap="small", vertical_alignment="bottom"
                )
                with col_text:
                    # Keyed input: this key IS the shared draft state that
                    # the suggested-prompt buttons write into (inserted
                    # text is immediately visible), and the on_click
                    # submit callback clears it after staging the message.
                    st.text_input(
                        "Message",
                        key=COMPOSER_INPUT_KEY,
                        disabled=disabled,
                        placeholder="Ask anything... (documents, tasks, or "
                        "general chat)",
                        label_visibility="collapsed",
                    )
                with col_send:
                    st.form_submit_button(
                        "Send",
                        key="pkta_composer_send",
                        disabled=disabled,
                        type="primary",
                        use_container_width=True,
                        on_click=_on_composer_submit,
                        args=(container, thread_id),
                    )
            # The callback staged a message to send (and cleared the box):
            # submit it now, from the main script body.
            pending_send = st.session_state.pop(COMPOSER_PENDING_SEND, None)
            if pending_send and not disabled:
                _submit_message(container, thread_id, pending_send)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _active_thread_id() -> str | None:
    return st.session_state.get("pkta_thread")


def _pending_user_text(messages, job) -> str | None:
    """The user bubble for a not-yet-persisted message.

    The worker writes the HumanMessage into the checkpoint almost
    immediately; after that the transcript already shows it and the
    pending bubble must not duplicate it (full reruns can happen at any
    point during generation - Stop clicks, sidebar navigation).
    """
    if job is None:
        return None
    last = messages[-1] if messages else None
    if (
        last is not None
        and last.__class__.__name__ == "HumanMessage"
        and str(last.content) == job.text
    ):
        return None
    return job.text


def render(container) -> None:
    thread_id = _active_thread_id()

    # Consume a job that finished while the user was elsewhere (another
    # page, another chat): its result lives in the checkpoint; only the
    # one-shot outcome notice needs to be surfaced here.
    stale_token = st.session_state.get("pkta_processing")
    if stale_token:
        stale_job = generation.get_job(stale_token)
        if stale_job is not None and stale_job.terminal:
            _consume_terminal_job(stale_token, stale_job)
            st.rerun()

    outcome = st.session_state.pop("pkta_last_outcome", None)
    if outcome and outcome.get("status") == generation.FAILED:
        st.error(
            "The request failed: "
            + (outcome.get("error") or "unknown error")
        )

    job, token = _active_job_for(thread_id)
    generating = job is not None and not job.terminal

    # -- conversation transcript ---------------------------------------------
    messages = get_thread_messages(container, thread_id) if thread_id else []
    pending_text = _pending_user_text(messages, job) if generating else None

    if not messages and not pending_text:
        _render_empty_state()
    else:
        for message in messages:
            render_message(message)
        if pending_text:
            # The submitted message is not in the checkpoint yet: show it
            # as a pending bubble; the monitor fragment paints the working
            # indicator right below.
            _render_user_message(HumanMessage(content=pending_text))

    if generating:
        # Live progress + completion handling for THIS thread's job.
        _generation_monitor(token)
    else:
        # No active generation on this thread: a queued message (submitted
        # while the previous stopped run was winding down) can start now.
        _maybe_start_queued(container, thread_id)

    # -- pending human approval (durable interrupt) ---------------------------
    approval_pending = False
    pending = pending_interrupt(container, thread_id) if thread_id else None
    if pending:
        approval_pending = True
        render_proposal_card(
            container,
            operation_id=pending["operation_id"],
            proposal=pending["proposal"],
            thread_id=thread_id,
            show_context=False,
        )

    # -- drag-and-drop upload across the chat/message area --------------------
    # Disabled while this thread is generating (ingestion is main-thread
    # work; the plus button and drops come back as soon as it finishes).
    from ui.dropzone import register_chat_dropzone

    register_chat_dropzone(
        container,
        disabled=generating or approval_pending,
        on_files=lambda pairs: _handle_dropped_files(container, pairs),
    )

    # -- composer: previews above, + left, input, Send/Stop right ------------
    _render_composer(container, thread_id, job, token, approval_pending)
