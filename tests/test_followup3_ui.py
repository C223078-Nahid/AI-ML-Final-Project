"""UI contracts for the follow-up task 3 (sections 2-5, 7-8).

Source-level canaries (the same pattern the existing suite uses for
Streamlit UI) plus pure-Python unit tests for the dropzone payload
decoder.  Browser-level verification (real rendering, click-through,
drag-drop) is done separately with the running app - see
scripts/ui-check/.
"""

from __future__ import annotations

import base64
import inspect


# ---------------------------------------------------------------------------
# Section 2: branding cleanup
# ---------------------------------------------------------------------------
def test_no_branding_text_anywhere_in_ui_modules():
    """The removed taglines must not reappear in any RENDERED UI string
    (docstrings may document what was removed)."""
    import app
    import ui.chat
    import ui.documents
    import ui.settings
    import ui.tasks
    import ui.approvals

    rendered_markers = (
        "st.markdown", "st.caption", "st.button", "st.write", "st.info",
        "st.warning", "st.subheader", "st.header", "st.text",
    )
    for module in (app, ui.chat, ui.documents, ui.settings, ui.tasks,
                   ui.approvals):
        source = inspect.getsource(module)
        rendered_lines = [
            line for line in source.splitlines()
            if any(marker in line for marker in rendered_markers)
        ]
        rendered = "\n".join(rendered_lines)
        assert "local · private · yours" not in rendered, (
            f"{module.__name__} renders the removed branding sub-line"
        )
        assert "RAG over your documents" not in rendered, (
            f"{module.__name__} renders the removed footer tagline"
        )


def test_sidebar_top_padding_is_small_but_present():
    """Follow-up task 4, section 3: the application name sits near the
    top-left with a SMALL amount of top spacing - present, but without the
    native Streamlit header spacer (never pressed against the edge)."""
    from ui.theme import THEME_CSS

    # The native sidebar header stays hidden (no 172px spacer).
    assert '[data-testid="stSidebarHeader"] {display: none;}' in (
        THEME_CSS.replace("\n", " ")
    ) or "[data-testid=\"stSidebarHeader\"] {display: none;}" in THEME_CSS
    # A small, deliberate top padding exists above the sidebar content.
    assert "padding-top: 0.7rem" in THEME_CSS
    # The brand row itself carries breathing room and a fixed height so
    # the toggle can vertically center against the name.
    assert "min-height: 2.2rem" in THEME_CSS
    assert ".pkta-brand" in THEME_CSS


def test_sidebar_toggle_is_vertically_centered_with_brand():
    """Follow-up task 4, section 3: the hamburger toggle is a fixed-height
    square centered on the brand row (same 2.2rem height, flex centering)."""
    from ui.theme import THEME_CSS

    assert ".st-key-pkta_sidebar_collapse button" in THEME_CSS
    assert "align-items: center" in THEME_CSS
    assert "height: 2.2rem" in THEME_CSS


def test_plus_icon_is_absolutely_centered():
    """Follow-up task 4, section 2: the + glyph is centered inside its
    button container with absolute positioning + translate(-50%, -50%) -"
    "dimension-independent centering."""
    from ui.theme import THEME_CSS

    selector = "[data-testid=\"stFileUploaderDropzone\"]::after"
    assert selector in THEME_CSS
    assert "transform: translate(-50%, -50%);" in THEME_CSS
    assert "top: 50%;" in THEME_CSS
    assert "left: 50%;" in THEME_CSS


def test_sidebar_left_alignment_contract():
    """All sidebar content is left-aligned (no centered elements)."""
    from ui.theme import THEME_CSS

    assert "text-align: left" in THEME_CSS  # brand + chats heading
    assert "justify-content: flex-start" in THEME_CSS  # brand row
    # Sidebar nav buttons were already left-aligned and stay that way.
    assert "text-align: left;" in THEME_CSS


def test_chat_history_scrolls_independently():
    """The chat history section is visually separated (top border on the
    Chats heading) and scrolls on its own."""
    import app
    from ui.theme import THEME_CSS

    source = inspect.getsource(app._render_sidebar)
    assert "pkta-history-scroll-start" in source
    assert "with st.container():" in source
    assert "overflow-y: auto" in THEME_CSS
    assert "border-top" in THEME_CSS  # visual separation of the Chats block


# ---------------------------------------------------------------------------
# Section 4: the plus button opens the file browser directly
# ---------------------------------------------------------------------------
def test_plus_button_is_direct_file_browser():
    """No popover, no caret: the composer's plus control is a collapsed
    st.file_uploader whose dropzone is styled as the compact square."""
    import ui.chat as chat_module

    source = inspect.getsource(chat_module._render_composer)
    assert "st.popover" not in source
    assert 'label_visibility="collapsed"' in source
    assert "type=SUPPORTED_TYPES" in source
    assert "accept_multiple_files=True" in source
    # One click opens the browser: the uploader sits in the plus column
    # BEFORE the form column.
    plus_index = source.index("col_plus")
    assert source.index("st.file_uploader") > plus_index
    assert source.index("st.file_uploader") < source.index("st.form(")


def test_plus_styling_hides_caret_and_shows_single_glyph():
    """The theme styles the composer uploader as ONE compact square with a
    single + glyph; the Documents-page uploader is unaffected."""
    from ui.theme import THEME_CSS

    assert 'content: "＋"' in THEME_CSS
    assert ".pkta-composer-bar) [data-testid=\"stFileUploaderDropzone\"]" in (
        THEME_CSS
    )


def test_selection_ingests_immediately_through_shared_pipeline():
    """Files picked through + are processed IMMEDIATELY (not staged as raw
    bytes until send), through the SAME process_upload pipeline used by the
    Documents page."""
    import ui.chat as chat_module

    handle = inspect.getsource(chat_module._handle_attachment_selection)
    ingest = inspect.getsource(chat_module._ingest_and_stage)
    assert "_ingest_and_stage" in handle
    assert "process_upload" in ingest
    assert "_stage_attachment" in ingest
    # The uploader key is bumped after selection so the widget resets.
    assert "pkta_uploader_seq" in handle


def test_attachment_chip_contract():
    """Each chip shows file-type badge, filename, indexing status, and a
    remove affordance; removal keeps the document in the library."""
    import ui.chat as chat_module

    source = inspect.getsource(chat_module._render_attachment_chips)
    assert "_file_icon" in source          # file-type badge
    assert "att['name']" in source         # filename
    assert "indexed" in source             # indexing status
    assert "st.button" in source           # the remove affordance
    help_text = inspect.getsource(chat_module._render_attachment_chips)
    assert "shared library" in help_text   # removal does not delete the doc


def test_chips_render_above_composer_input():
    import ui.chat as chat_module

    source = inspect.getsource(chat_module._render_composer)
    assert source.index("_render_attachment_chips") < source.index(
        "st.file_uploader"
    ) < source.index("st.form(")


# ---------------------------------------------------------------------------
# Section 5: drag-and-drop across the chat area
# ---------------------------------------------------------------------------
def test_dropzone_component_mounted_on_chat_page():
    import ui.chat as chat_module

    source = inspect.getsource(chat_module.render)
    assert "register_chat_dropzone" in source
    assert "disabled=generating or approval_pending" in source


def test_dropzone_frontend_exists_and_speaks_component_protocol():
    from pathlib import Path

    frontend = (
        Path(__file__).resolve().parent.parent
        / "ui" / "components" / "pkta_dropzone" / "index.html"
    )
    assert frontend.exists(), "the dropzone component frontend is missing"
    js = frontend.read_text(encoding="utf-8")
    assert "streamlit:componentReady" in js
    assert "streamlit:setComponentValue" in js
    assert "streamlit:setFrameHeight" in js
    # Drops aimed at native uploaders are left to Streamlit (no double
    # processing).
    assert "stFileUploaderDropzone" in js
    # The drag-over overlay shows a clear instruction.
    assert "Drop to upload" in js


def test_dropzone_payload_decoding_and_validation():
    """The Python side decodes base64 payloads and validates types/sizes."""
    from ui.dropzone import _decode_drop_payload

    good = base64.b64decode
    payload = {
        "nonce": "n-1",
        "files": [
            {"name": "notes.txt", "type": "text/plain", "size": 6,
             "data": base64.b64encode(b"hello\n").decode()},
            {"name": "paper.md", "type": "", "size": 2,
             "data": base64.b64encode(b"# ").decode()},
        ],
    }
    decoded = _decode_drop_payload(payload)
    assert decoded == [("notes.txt", b"hello\n"), ("paper.md", b"# ")]

    assert _decode_drop_payload(None) is None
    assert _decode_drop_payload({"files": "nope"}) is None
    assert _decode_drop_payload({"nonce": "n", "files": []}) == []
    # Malformed base64 entries are skipped, not fatal.
    mixed = {
        "nonce": "n-2",
        "files": [
            {"name": "bad.txt", "data": "!!!not-base64!!!"},
            {"name": "ok.txt", "data": base64.b64encode(b"x").decode()},
        ],
    }
    assert _decode_drop_payload(mixed) == [("ok.txt", b"x")]


def test_dropzone_supported_extensions_match_composer():
    import ui.chat as chat_module
    from ui.dropzone import SUPPORTED_EXTENSIONS

    assert SUPPORTED_EXTENSIONS == {".pdf", ".docx", ".txt", ".md"}
    assert set(chat_module.SUPPORTED_TYPES) == {"pdf", "docx", "md", "txt"}


# ---------------------------------------------------------------------------
# Sections 7-8: Stop button and progress phases
# ---------------------------------------------------------------------------
def test_stop_button_uses_stop_icon_and_signals_cancellation():
    import ui.chat as chat_module

    source = inspect.getsource(chat_module._render_composer)
    assert 'icon=":material/stop:"' in source
    assert "generation.request_stop" in source
    # The input shows the generating placeholder while running.
    assert "Generating response..." in source


def test_progress_labels_are_real_application_events():
    """Every phase label the UI can display originates from an actual
    report_phase call in the workflow nodes (never fabricated)."""
    import graph.nodes as nodes_module

    source = inspect.getsource(nodes_module)
    for label in (
        "Searching uploaded documents…",
        "Retrieving relevant passages…",
        "Preparing response…",
        "Thinking…",
        "Stopping…",
    ):
        assert label in source, f"phase label {label!r} not reported by nodes"


def test_monitor_shows_real_phase_and_stopping_state():
    import ui.chat as chat_module

    source = inspect.getsource(chat_module._generation_monitor)
    assert "current_phase" in source
    assert "Stopping…" in source
    assert "cancel_event.is_set()" in source
    assert "run_every" in source


# ---------------------------------------------------------------------------
# Section 6: the Settings page surfaces the provider chain
# ---------------------------------------------------------------------------
def test_settings_page_shows_provider_chain_and_cloudflare_status():
    import ui.settings as settings_module

    source = inspect.getsource(settings_module._render_status)
    assert 'gateway_info["chain"]' in source
    assert "cloudflare" in source.lower()
    assert "LLM_FALLBACK_PROVIDERS" in source
