"""Python side of the chat drag-and-drop component (follow-up task, section 5).

The frontend (``ui/components/pkta_dropzone/index.html``) attaches
document-level drag listeners to the Streamlit app (same-origin iframe) and
sends dropped files - base64-encoded, with a per-drop nonce - back through
the standard component-value channel.  This module:

- registers the component exactly once per process,
- consumes each drop exactly once (nonce guard against the component value
  persisting across reruns),
- validates type and size before anything reaches the ingestion pipeline,
- and hands valid files to the SAME upload entry point as the plus button
  and the Documents page (no second ingestion implementation).
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "pkta_dropzone"

_component = components.declare_component("pkta_dropzone", path=str(_COMPONENT_DIR))

#: Kept in sync with ui/chat.SUPPORTED_TYPES (single source of truth there;
#: this is the validation copy for dropped files).
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _decode_drop_payload(payload: Any) -> list[tuple[str, bytes]] | None:
    """Validate and decode a component payload into (name, bytes) pairs.

    Returns None when the payload is malformed; individual invalid files are
    skipped, and the caller reports them.
    """
    if not isinstance(payload, dict):
        return None
    files = payload.get("files")
    if not isinstance(files, list):
        return None
    decoded: list[tuple[str, bytes]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        data = str(item.get("data") or "")
        if not name or not data:
            continue
        try:
            decoded.append((name, base64.b64decode(data, validate=True)))
        except (binascii.Error, ValueError):
            continue
    return decoded


def register_chat_dropzone(container, *, disabled: bool = False,
                           on_files=None) -> None:
    """Mount the invisible dropzone listener on the chat page.

    ``on_files`` receives a list of (filename, bytes) pairs and is
    responsible for the actual processing (the chat page passes a callback
    that runs the shared ingestion pipeline and stages attachment chips).
    Dropped files that fail validation are reported with st.warning, never
    silently dropped.
    """
    payload = _component(default=None, key="pkta_dropzone", disabled=disabled)
    if payload is None:
        return
    # Consume each drop exactly once: the component value persists across
    # reruns, so the nonce (fresh per drop event) gates re-processing.
    nonce = payload.get("nonce") if isinstance(payload, dict) else None
    if not nonce or st.session_state.get("pkta_dropzone_nonce") == nonce:
        return
    st.session_state["pkta_dropzone_nonce"] = nonce

    decoded = _decode_drop_payload(payload)
    if not decoded:
        st.warning(
            "The dropped item could not be read as a file. Supported "
            "formats: PDF, DOCX, TXT, Markdown."
        )
        return

    max_bytes = container.settings.max_file_size_bytes
    accepted: list[tuple[str, bytes]] = []
    rejected: list[str] = []
    for name, data in decoded:
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            rejected.append(f"{name} (unsupported format)")
        elif len(data) > max_bytes:
            rejected.append(
                f"{name} ({len(data) / 1024 / 1024:.1f} MB exceeds the "
                f"{container.settings.max_file_size_mb} MB limit)"
            )
        elif not data:
            rejected.append(f"{name} (empty file)")
        else:
            accepted.append((name, data))

    for reason in rejected:
        st.warning(f"Skipped {reason}.")

    if accepted and on_files is not None:
        on_files(accepted)
