"""Document Library page: upload, indexing status, and removal.

Every uploaded document goes into ONE shared library: a single Qdrant
collection that every chat session queries. Removing a document removes it
from that shared index for all chats.
"""

from __future__ import annotations

import html

import streamlit as st

from core import guardrails
from core.models import DocumentRecord
from ui.uploads import process_upload_list

SUPPORTED_TYPES = ["pdf", "docx", "md", "txt"]


def _status_badge(status: str) -> str:
    colors = {"indexed": "#10a37f", "processing": "#e0a82e", "error": "#ef6c57"}
    labels = {"indexed": "indexed", "processing": "working", "error": "error"}
    color = colors.get(status, "#9b9b9b")
    label = labels.get(status, status)
    return (
        f'<span style="color:{color};font-size:0.75rem;font-weight:600;">'
        f"● {label}</span>"
    )


def _render_document_row(container, record: DocumentRecord) -> None:
    with st.container():
        col_info, col_meta, col_action = st.columns([4.2, 3.4, 1.2])
        with col_info:
            st.markdown(f"**{html.escape(record.filename)}**")
            st.caption(
                f"{record.file_type} · uploaded "
                f"{record.uploaded_at[:16].replace('T', ' ')}"
            )
        with col_meta:
            badge = _status_badge(record.status)
            if record.status == "indexed":
                pages = (
                    f" · {record.page_count} pages" if record.page_count else ""
                )
                st.markdown(
                    f"{badge} · {record.chunk_count} chunks{pages}",
                    unsafe_allow_html=True,
                )
            elif record.status == "error":
                st.markdown(badge, unsafe_allow_html=True)
                st.caption(record.error_message or "Unknown error")
            else:
                st.markdown(badge, unsafe_allow_html=True)
        with col_action:
            with st.popover("Remove ⌄", use_container_width=True):
                st.write(f"Remove **{html.escape(record.filename)}** from the "
                         "shared library?")
                st.caption(
                    "The document becomes unavailable to every chat. This "
                    "cannot be undone."
                )
                if st.button("Confirm removal", key=f"del-{record.document_id}",
                             type="primary"):
                    container.ingestion.delete_document(record.document_id)
                    st.rerun()


def render(container) -> None:
    # Page title/subtitle are rendered by the application header (app.py).
    settings = container.settings
    documents = container.doc_repo.list_documents()
    active = guardrails.validate_document_count(documents)
    st.caption(
        f"{active} of {settings.max_documents} documents indexed · "
        f"max {settings.max_file_size_mb} MB per file · "
        f"supported: PDF, DOCX, Markdown, TXT"
    )

    upload = st.file_uploader(
        "Drop documents here or click to browse",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="pkta_documents_uploader",
    )
    if upload:
        process_upload_list(container, upload)

    st.divider()
    if not documents:
        st.info(
            "No documents yet. Upload a PDF, DOCX, Markdown, or TXT file to "
            "build your personal knowledge base — it will be available in "
            "every conversation."
        )
        return
    for record in documents:
        _render_document_row(container, record)
