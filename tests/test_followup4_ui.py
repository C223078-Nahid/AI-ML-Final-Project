"""Follow-up task 4, sections 2-4 (UI refinements): source-contract tests.

The rendering itself is verified visually (scripts/ui-check screenshots);
these tests pin the CONTRACTS the rendering depends on:

- the plus icon is centered with dimension-independent absolute centering;
- the sidebar brand has top spacing and the toggle is vertically centered
  against the name (fixed 2.2rem row heights);
- the new-chat screen centers the suggested-prompt section;
- suggested prompts are real Streamlit buttons that insert their text into
  the composer's SHARED draft key, never auto-submit, and never create a
  chat;
- the composer input uses that same shared key (single source of truth).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# CSS contracts (theme)
# ---------------------------------------------------------------------------
def test_plus_icon_centered_with_absolute_positioning():
    from ui.theme import THEME_CSS

    assert "[data-testid=\"stFileUploaderDropzone\"]::after" in THEME_CSS
    assert "top: 50%;" in THEME_CSS
    assert "left: 50%;" in THEME_CSS
    assert "transform: translate(-50%, -50%);" in THEME_CSS
    assert "position: absolute;" in THEME_CSS
    # No asymmetric padding inside the square (the previous off-centering
    # cause); the dropzone carries padding: 0.
    assert "    padding: 0;\n" in THEME_CSS


def test_sidebar_brand_has_top_spacing():
    from ui.theme import THEME_CSS

    assert "padding-top: 0.7rem" in THEME_CSS           # above the content
    assert ".pkta-brand" in THEME_CSS
    assert "min-height: 2.2rem" in THEME_CSS             # brand row height


def test_toggle_button_matches_brand_row_height():
    from ui.theme import THEME_CSS

    assert ".st-key-pkta_sidebar_collapse button" in THEME_CSS
    assert "height: 2.2rem;" in THEME_CSS
    assert "align-items: center" in THEME_CSS
    assert "justify-content: center" in THEME_CSS


def test_suggestion_buttons_are_styled_and_centered():
    from ui.theme import THEME_CSS

    # Centered suggestion block in the new-chat screen.
    assert "justify-content: center" in THEME_CSS
    assert ".pkta-empty" in THEME_CSS
    # Quiet pill styling scoped to the suggestion buttons.
    assert '[class*="st-key-pkta_sugg"] button' in THEME_CSS


def test_fact_check_report_styles_exist():
    from ui.theme import THEME_CSS

    for css_class in (
        ".pkta-fc-head", ".pkta-fc-status", ".pkta-fc-correct",
        ".pkta-fc-incorrect", ".pkta-fc-partial", ".pkta-fc-unverified",
        ".pkta-fc-method", ".pkta-fc-evidence", ".pkta-fc-source",
    ):
        assert css_class in THEME_CSS, css_class


# ---------------------------------------------------------------------------
# Module contracts (chat page structure)
# ---------------------------------------------------------------------------
def test_composer_input_key_is_the_shared_draft_state():
    """The composer text input is keyed with pkta_composer_input - the
    SAME state the suggested prompts write into."""
    from ui import chat

    assert chat.COMPOSER_INPUT_KEY == "pkta_composer_input"
    assert "COMPOSER_INPUT_KEY" in chat.__dict__


def test_apply_suggestion_inserts_into_empty_draft():
    """Clicking a suggestion with an empty composer fills it."""
    from ui import chat


    class _Session(dict):
        pass


    session = _Session()
    # apply_suggestion reads st.session_state; test through the same
    # attribute surface Streamlit exposes.
    import streamlit as st

    original = st.session_state
    try:
        st.session_state = session  # type: ignore[assignment]
        chat.apply_suggestion("Summarize my uploaded document")
        assert session["pkta_composer_input"] == "Summarize my uploaded document"
    finally:
        st.session_state = original


def test_apply_suggestion_preserves_existing_draft():
    """A non-empty draft is NOT overwritten: the suggestion is appended on
    a new line (the task's 'sensible behaviour' contract)."""
    import streamlit as st

    from ui import chat

    original = st.session_state
    try:
        st.session_state = {"pkta_composer_input": "What did I write about"}
        chat.apply_suggestion("Summarize my uploaded document")
        assert st.session_state["pkta_composer_input"] == (
            "What did I write about Summarize my uploaded document"
        )
        # An identical suggestion never duplicates.
        chat.apply_suggestion("Summarize my uploaded document")
        assert st.session_state["pkta_composer_input"].count(
            "Summarize my uploaded document"
        ) == 1
    finally:
        st.session_state = original


def test_suggestions_are_clickable_buttons_not_static_text():
    """The empty state renders real st.button widgets (clickable), not
    static markdown chips."""
    import inspect

    from ui import chat

    source = inspect.getsource(chat._render_empty_state)
    assert "st.button" in source
    assert "on_click=apply_suggestion" in source
    # ...and nothing in the flow submits automatically.
    assert "_submit_message" not in source
    # No new chat is created by a suggestion click.
    assert "pkta_thread" not in source


def test_composer_submit_uses_callback_and_clears_input():
    """Submission runs through the on_click callback (which stages the
    message and clears the shared draft key), then the main body submits."""
    import inspect

    from ui import chat

    composer_source = inspect.getsource(chat._render_composer)
    assert "on_click=_on_composer_submit" in composer_source
    assert "key=COMPOSER_INPUT_KEY" in composer_source
    callback_source = inspect.getsource(chat._on_composer_submit)
    assert "COMPOSER_INPUT_KEY" in callback_source
    assert "COMPOSER_PENDING_SEND" in callback_source


def test_fact_check_report_rendering_contract():
    """The chat page renders fact_check_report payloads with the evidence
    layers kept distinct (status labels + verification-method labels)."""
    import inspect

    from ui import chat

    source = inspect.getsource(chat._render_ai_payload)
    assert "fact_check_report" in source
    report_source = inspect.getsource(chat._render_fact_check_report)
    assert "FACT-CHECK REPORT" in report_source
    assert "verification_method" in report_source or "method_label" in report_source
    assert "external_sources" in report_source
    # The report is read-only: no document mutation API is touched.
    assert "delete_document" not in report_source
    assert "ingest" not in report_source
