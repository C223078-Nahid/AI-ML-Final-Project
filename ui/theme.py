"""ChatGPT-inspired dark theme for the Streamlit UI.

Design goals:
- Dark, minimal, professional: quiet background, subtle borders, rounded
  components, clear hierarchy, no decorative clutter.
- Text is never clipped: flexible widths, ``min-width: 0`` on flex children,
  and ``overflow-wrap`` everywhere (fixes the Settings truncation bug where
  labels such as "return" were displayed as "retur").
- The layout adapts to narrower windows; the sidebar and composer never
  overlap the content.

Sidebar controls (follow-up bug fix - "sidebar cannot be reopened"):
- ROOT CAUSE: Streamlit renders the native sidebar-expand button INSIDE the
  app ``header`` element. Hiding the header with
  ``header {visibility: hidden}`` therefore also hid the expand button
  (``visibility`` inherits), so a collapsed sidebar had no visible way back.
- Fix: keep hiding the app chrome, but force the native expand control
  visible again, and hide the native collapse control inside the sidebar so
  the application-level toggle (app.py) is the single source of truth.
"""

from __future__ import annotations

import streamlit as st

THEME_CSS = """
<style>
/* ------------------------------------------------------------------ base */
.stApp {
    background-color: #212121;
    color: #ececec;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 52rem;}

/* Anti-truncation: nothing may hard-clip text in any container. */
.stApp, .stApp * {
    overflow-wrap: break-word;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] label {
    overflow-wrap: anywhere;
    white-space: normal;
}
[data-testid="stMarkdownContainer"] code,
[data-testid="stMarkdownContainer"] pre,
code, pre {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
/* Flex children must be allowed to shrink so long words wrap instead of
   pushing siblings out of view (the root cause of clipped labels). */
[data-testid="stHorizontalBlock"] > div,
[data-testid="stColumn"] {min-width: 0;}

/* ------------------------------------------------------------- sidebar */
/* Left-aligned sidebar content; the application name sits as close to
   the top-left as the layout permits (follow-up task section 3).

   Streamlit 1.67 renders a ~172px native sidebar HEADER (logo spacer +
   native collapse control) ABOVE the app content; hiding it removes the
   unnecessary top padding - the application-level toggle owns
   collapse/restore, so nothing is lost. */
[data-testid="stSidebar"] {
    background-color: #171717;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebar"] * {font-size: 0.92rem;}
[data-testid="stSidebarHeader"] {display: none;}
/* Captions must stay readable on the dark sidebar. */
[data-testid="stSidebar"] p {color: #c2c2c2;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #c2c2c2;}
/* Full-height flex column: brand / New Chat / navigation pinned at the
   top, chat history flexing to the remaining height (and scrolling on its
   own - see the history rules below). */
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarUserContent"] > div {
    display: flex; flex-direction: column;
    flex: 1 1 auto; min-height: 0;
}
/* Follow-up task 4, section 3: a small amount of top spacing above the
   application name so it sits near the top-left without being pressed
   against the sidebar's top edge. */
[data-testid="stSidebarUserContent"] > div {padding-top: 0.7rem;}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    display: flex; flex-direction: column;
    flex: 1 1 auto; min-height: 0;
}

/* --- Sidebar toggle controls (reopen-bug fix) ------------------------- */
/* 1. ROOT CAUSE FIX: the native expand control renders inside <header>;
    visibility would inherit "hidden" from the hidden app chrome. Keep it
    visible by default so a collapsed sidebar always has a native way back
    (app.py additionally hides the native controls because the app-header
    toggle is the single source of truth for visibility). */
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    color: #b9b9b9;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}
[data-testid="stExpandSidebarButton"]:hover {color: #ffffff;}
/* 2. The native collapse control inside the sidebar is hidden: the app
    header toggle (app.py) owns collapse/restore so there is exactly one
    source of truth for sidebar visibility. */
[data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
[data-testid="stSidebar"] button[aria-label="Close sidebar"] {display: none;}
.pkta-brand {
    display: flex; align-items: center; gap: 0.55rem;
    /* Follow-up task 4, section 3: the brand row is EXACTLY the toggle's
       height (2.2rem) so the name's vertical center coincides with the
       toggle's; the small top spacing comes from the sidebar content
       padding above this row. */
    height: 2.2rem;
    padding: 0 0.35rem;
    margin: 0;
    box-sizing: border-box;
    justify-content: flex-start;
    text-align: left;
}
/* The brand/toggle columns row: center both columns vertically (the brand
   column is the taller one, so the toggle centers exactly on the name).
   Follow-up task 4, section 3 - vertically center the sidebar toggle with
   the application name. */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="pkta_sidebar_collapse"]) {
    align-items: center;
}
/* Streamlit's markdown wrapper chain collapses to the text line-height;
   force it to hand the .pkta-brand box through untouched. */
[data-testid="stSidebar"] [data-testid="stMarkdown"]:has(.pkta-brand),
[data-testid="stSidebar"] [data-testid="stMarkdown"] > div:has(.pkta-brand),
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.pkta-brand) {
    height: 2.2rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] > div:has(.pkta-brand) {
    display: block;
}
/* Breathing room between the brand row and the New Chat button (moved
   here from the brand's old bottom margin). */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="pkta_sidebar_collapse"]) + [data-testid="stButton"] {
    margin-top: 0.45rem;
}
.pkta-brand .pkta-logo {
    width: 30px; height: 30px; border-radius: 9px;
    background: #10a37f; color: #04110c;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px;
    flex: 0 0 auto;
}
.pkta-brand .pkta-name {font-weight: 650; font-size: 1.0rem; color: #f3f3f3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}

/* Sidebar chat history section: visually separated from the application
   navigation and independently scrollable when the list grows long.
   (Streamlit renders st.container() as stLayoutWrapper > stVerticalBlock,
   with every element wrapped in stElementContainer > stMarkdown - the
   anchored :has() selectors below target exactly that container.) */
.pkta-history {margin-top: 0.35rem;}
.pkta-history-title {
    font-size: 0.72rem; color: #9b9b9b; text-transform: uppercase;
    letter-spacing: 0.05em; padding: 0.7rem 0.45rem 0.3rem 0.45rem;
    text-align: left; border-top: 1px solid rgba(255, 255, 255, 0.07);
    margin-top: 0.4rem;
}
[data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .pkta-history-scroll-start) {
    display: flex; flex-direction: column;
    flex: 1 1 auto; min-height: 3.5rem;
}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-history-scroll-start) {
    flex: 1 1 auto;
    overflow-y: auto;
    min-height: 3.5rem;
    scrollbar-width: thin;
    padding-right: 0.15rem;
}

/* Sidebar nav (secondary buttons styled flat and quiet, LEFT-ALIGNED:
   Streamlit 1.67 buttons are inline-flex with centered content, so the
   left alignment must be forced with justify-content, not text-align). */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px;
    border: 1px solid transparent;
    color: #d5d5d5;
    background: transparent;
    text-align: left;
    justify-content: flex-start;
    padding: 0.32rem 0.55rem;
    font-weight: 450;
    transition: background 0.15s ease;
}
/* Streamlit 1.67 wraps the label in an inner flex div with centered
   content - the left alignment must be forced there as well. */
[data-testid="stSidebar"] .stButton > button > div {
    justify-content: flex-start;
    text-align: left;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: transparent;
    color: #ffffff;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2f2f2f;
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.09);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #3a3a3a;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
    gap: 0.25rem;
}

/* ---------------------------------------------------------- main area */
/* Page-title header (slim row above the page content; the sidebar toggle
   lives on the sidebar side, not here). */
.pkta-page-header {
    padding: 0.25rem 0.35rem 0.45rem 0.35rem;
    font-size: 1.05rem; font-weight: 600; color: #f3f3f3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pkta-page-sub {
    font-size: 0.8rem; color: #9b9b9b; margin-top: 0.1rem;
    white-space: normal; overflow-wrap: anywhere;
}

/* --- Collapsed-sidebar reopen rail (left edge of the application) ------
   The rail row renders a pkta-rail-marker div; :has() pins that row to a
   fixed top-left position (the same corner the sidebar toggle occupies
   when the sidebar is open) and indents the page title so nothing ever
   overlaps on narrow windows. */
div[data-testid="stHorizontalBlock"]:has(.pkta-rail-marker) {
    position: fixed;
    top: 0.55rem;
    left: 0.55rem;
    z-index: 999;
    width: auto;
    margin: 0 !important;
    gap: 0 !important;
}
/* The rail's narrow column must size to its button, not to a flex ratio of
   a zero-width fixed row (otherwise the button overflows off-screen). */
div[data-testid="stHorizontalBlock"]:has(.pkta-rail-marker) > div,
div[data-testid="stHorizontalBlock"]:has(.pkta-rail-marker) [data-testid="stColumn"] {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
.stApp:has(.pkta-rail-marker) .pkta-page-header {padding-left: 2.6rem;}
.st-key-pkta_sidebar_reopen button {
    border: 1px solid transparent;
    background: rgba(255, 255, 255, 0.05);
    color: #d5d5d5;
    font-size: 1.25rem;
    line-height: 1.2;
    padding: 0.05rem 0.15rem;
    min-height: 2.2rem;
    min-width: 2.2rem;
    border-radius: 10px;
    transition: background 0.15s ease;
}
.st-key-pkta_sidebar_reopen button:hover,
.st-key-pkta_sidebar_reopen button:focus {
    background: rgba(255, 255, 255, 0.12);
    color: #ffffff;
    border-color: transparent;
}

/* Sidebar-side toggle (top-left of the sidebar, next to the brand).
   Follow-up task 4, section 3: vertically centered with the application
   name - the toggle column uses vertical_alignment="center" and the
   button is a fixed 2.2rem square, matching the .pkta-brand row height. */
.st-key-pkta_sidebar_collapse button {
    border: 1px solid transparent;
    background: transparent;
    color: #d5d5d5;
    font-size: 1.25rem;
    line-height: 1;
    padding: 0;
    min-height: 2.2rem;
    min-width: 2.2rem;
    height: 2.2rem;
    width: 2.2rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s ease;
}
.st-key-pkta_sidebar_collapse button:hover,
.st-key-pkta_sidebar_collapse button:focus {
    background: rgba(255, 255, 255, 0.08);
    color: #ffffff;
    border-color: transparent;
}

/* Chat history area */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.1rem 0;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    padding: 0.05rem 0.1rem;
}
/* Assistant avatar: small, quiet */
[data-testid="stChatMessage"] img, [data-testid="stChatMessage"] svg {
    width: 26px !important; height: 26px !important;
    border-radius: 7px;
}

/* User message: right-aligned bubble (rendered via pkta-user-bubble) */
.pkta-user-row {
    display: flex; justify-content: flex-end;
    padding: 0.45rem 0.9rem;
}
.pkta-user-bubble {
    background: #2f2f2f;
    color: #ececec;
    padding: 0.6rem 0.95rem;
    border-radius: 16px;
    max-width: 78%;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 0.95rem;
    line-height: 1.45;
    border: 1px solid rgba(255, 255, 255, 0.04);
}

/* Empty-state welcome (follow-up task 4, section 4: the whole new-chat
   section - heading, subtitle, and suggested prompts - is centered in the
   main chat area). */
.pkta-empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center;
    padding: 3.2rem 1rem 1.4rem 1rem;
}
.pkta-empty .pkta-hello {font-size: 1.55rem; font-weight: 650; color: #f3f3f3;}
.pkta-empty .pkta-subtitle {color: #9b9b9b; margin-top: 0.35rem; font-size: 0.95rem;}
.pkta-suggestions {margin-top: 1.4rem; display: flex; flex-wrap: wrap;
    gap: 0.55rem; justify-content: center; max-width: 34rem;}
/* Suggested-prompt buttons (follow-up task 4, section 4): quiet centered
   pills - restrained styling, no card chrome, clearly clickable. */
[class*="st-key-pkta_sugg"] button {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 12px;
    color: #c9c9c9;
    font-size: 0.82rem;
    font-weight: 450;
    padding: 0.42rem 0.6rem;
    transition: background 0.15s ease, border-color 0.15s ease;
}
[class*="st-key-pkta_sugg"] button:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.3);
    color: #f3f3f3;
}
[class*="st-key-pkta_sugg"] button:focus:not(:active) {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.3);
    color: #f3f3f3;
}

/* Fact-check report (follow-up task 4, section 10): restrained status
   labels, evidence layers kept visually distinct, read-only. */
.pkta-fc-head {
    margin-top: 0.55rem; margin-bottom: 0.3rem;
    font-size: 0.78rem; font-weight: 650; letter-spacing: 0.06em;
    color: #9b9b9b; text-transform: uppercase;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    padding-top: 0.55rem;
}
.pkta-fc-status {
    display: inline-block; font-size: 0.7rem; font-weight: 650;
    letter-spacing: 0.05em; border-radius: 6px; padding: 0.1rem 0.5rem;
    margin-right: 0.45rem;
}
.pkta-fc-correct {background: rgba(16, 163, 127, 0.16); color: #35d0a7;}
.pkta-fc-incorrect {background: rgba(224, 78, 78, 0.16); color: #ff8080;}
.pkta-fc-partial {background: rgba(230, 170, 40, 0.16); color: #eec06e;}
.pkta-fc-unverified {background: rgba(255, 255, 255, 0.08); color: #b9b9b9;}
.pkta-fc-method {
    display: inline-block; font-size: 0.7rem; color: #9b9b9b;
    border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 999px;
    padding: 0.08rem 0.5rem;
}
.pkta-fc-evidence {
    border-left: 3px solid rgba(16, 163, 127, 0.4);
    background: rgba(255, 255, 255, 0.02);
    padding: 0.35rem 0.6rem; margin: 0.3rem 0; font-size: 0.85rem;
    color: #c2c2c2; border-radius: 0 8px 8px 0;
}
.pkta-fc-source {
    font-size: 0.82rem; margin-top: 0.25rem;
}
.pkta-fc-source a {color: #6ec6ff; text-decoration: none; overflow-wrap: anywhere;}
.pkta-fc-source a:hover {text-decoration: underline;}
.pkta-fc-excerpt {
    color: #9b9b9b; font-size: 0.78rem; margin-top: 0.12rem;
    border-left: 2px solid rgba(255, 255, 255, 0.12);
    padding-left: 0.5rem;
}

/* Citations */
.pkta-sources {margin-top: 0.55rem;}
.pkta-source-item {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    margin-top: 0.4rem;
    background: rgba(255, 255, 255, 0.02);
    font-size: 0.85rem;
}
.pkta-source-item .pkta-source-head {color: #c9a8ff; font-weight: 570;}
.pkta-source-item .pkta-passage {color: #b9b9b9; margin-top: 0.25rem;
    font-size: 0.8rem; white-space: pre-wrap;}
.pkta-source-item .pkta-meta {color: #7d7d7d; font-size: 0.7rem; margin-top: 0.2rem;}

/* Tool trace chips */
.pkta-tool-trace {
    display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.45rem;
}
.pkta-tool-chip {
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 999px;
    color: #9b9b9b;
    font-size: 0.7rem;
    padding: 0.14rem 0.6rem;
}

/* Composer (follow-up task section 7): attachments above, + left,
   input middle, Send right. The composer container sticks to the bottom of
   the viewport so it stays reachable in long conversations. */
div[data-testid="stVerticalBlock"]:has(.pkta-composer-bar) {
    position: sticky;
    bottom: 0;
    z-index: 50;
    background: #212121;
    padding-bottom: 0.4rem;
}
.pkta-composer-bar {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.55rem 0.9rem 0.15rem 0.9rem;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 0.55rem;
}
/* The + attachment control (follow-up task 4, section 2): the collapsed
   file uploader rendered as ONE compact square button.  Clicking the
   dropzone opens the file browser directly - no popover, no caret.
   Scoped via the composer-bar marker so ONLY the composer's uploader is
   affected (the Documents-page uploader keeps its normal appearance). */
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"] {
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    min-height: 2.35rem;
    height: 2.35rem;
    width: 2.35rem;
    max-width: 2.35rem;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0;           /* hide the native "Browse files"/hint text */
    position: relative;
    flex: 0 0 auto;
    box-sizing: border-box;
}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"] > * {margin: auto;}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"] button {
    /* Contain the native "upload / Upload" button inside the square: it
       stays clickable (covering the whole square) but renders nothing. */
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    min-height: 0 !important;
    padding: 0 !important; margin: 0 !important;
    border: none !important;
    background: transparent !important;
    font-size: 0 !important;
    color: transparent !important;
    overflow: hidden;
    z-index: 0;
}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"] button * {
    visibility: hidden;
    font-size: 0 !important;
}
/* The visible "＋" glyph (painted above the invisible native button).
   Follow-up task 4, section 2 - the icon is centered EXACTLY inside the
   button container with absolute positioning + translate(-50%, -50%), so
   it stays perfectly centered no matter what in-flow children the native
   uploader renders and at any button dimension (the 900px media query
   shrinks the square to 2.1rem and the glyph recenters automatically). */
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"]::after {
    content: "＋";
    font-size: 1.15rem;
    font-weight: 600;
    color: #ececec;
    line-height: 1;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
    pointer-events: none;
}
/* The uploader's help-tooltip target stays out of the compact square. */
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploader"] [data-testid="stTooltipHoverTarget"],
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploader"] ~ [data-testid="stTooltipHoverTarget"] {
    display: none;
}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #10a37f;
    background: rgba(16, 163, 127, 0.08);
}

/* The composer's uploader container adds no extra chrome: the widget is
   exactly the compact square. */
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploader"] {
    margin: 0;
}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploader"] > div {
    min-height: 0;
}
/* Narrow windows: keep the plus square clear of the input. */
@media (max-width: 900px) {
    div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMarkdown"] .pkta-composer-bar) [data-testid="stFileUploaderDropzone"] {
        width: 2.1rem; height: 2.1rem; min-height: 2.1rem;
    }
}
\n/* The message input and the Send button share one row (inside the form). */
[data-testid="stForm"] input {
    background: #2f2f2f;
    color: #ececec;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stForm"] {background: transparent;}
.st-key-pkta_composer_send button {
    min-height: 2.35rem;
    border-radius: 12px;
    font-weight: 600;
}
/* The Stop button (follow-up task, section 7): replaces Send in the same
   position while the assistant is generating. */
.st-key-pkta_composer_stop button {
    min-height: 2.35rem;
    border-radius: 12px;
    font-weight: 600;
    background: #2f2f2f;
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: #ececec;
}
.st-key-pkta_composer_stop button:hover {
    background: #3a3a3a;
    border-color: rgba(255, 255, 255, 0.3);
    color: #ffffff;
}
/* Stopped-response marker under a cancelled assistant message. */
.pkta-stopped-marker {
    display: inline-flex; align-items: center; gap: 0.4rem;
    margin-top: 0.25rem;
    font-size: 0.75rem;
    color: #b9b9b9;
    border: 1px dashed rgba(255, 255, 255, 0.25);
    border-radius: 999px;
    padding: 0.12rem 0.6rem;
}
/* Attachment preview chips: quiet, rounded, with the remove affordance
   built into the button label ("✕ name"). */
.st-key-pkta_attach_remove button, [class*="st-key-pkta_attach_remove"] button {
    background: rgba(255, 255, 255, 0.04);
    border: 1px dashed rgba(255, 255, 255, 0.25);
    color: #cfcfcf;
    border-radius: 10px;
    font-size: 0.8rem;
    padding: 0.25rem 0.45rem;
}
.st-key-pkta_attach_remove button:hover, [class*="st-key-pkta_attach_remove"] button:hover {
    border-color: #ff6b6b;
    color: #ffffff;
}

/* Thinking indicator (follow-up task section 8): three pulsing dots plus
   a status label, rendered inside the assistant message bubble. */
.pkta-thinking {
    display: flex; align-items: center; gap: 0.5rem;
    color: #9b9b9b; font-size: 0.9rem; padding: 0.2rem 0.1rem 0.35rem 0.1rem;
}
.pkta-thinking-dots {display: inline-flex; gap: 4px;}
.pkta-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10a37f;
    display: inline-block;
    animation: pkta-pulse 1.1s ease-in-out infinite;
}
.pkta-dot:nth-child(2) {animation-delay: 0.18s;}
.pkta-dot:nth-child(3) {animation-delay: 0.36s;}
@keyframes pkta-pulse {
    0%, 100% {opacity: 0.25; transform: scale(0.85);}
    50% {opacity: 1.0; transform: scale(1.05);}
}

/* File uploader drop zone (drag-and-drop) */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1.5px dashed rgba(255, 255, 255, 0.22);
    border-radius: 14px;
    color: #b9b9b9;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"] > div > div:hover {
    border-color: #10a37f;
    color: #ececec;
}
/* Hide Streamlit's built-in "200MB per file" hint - the application's own
   limit (MAX_FILE_SIZE_MB, default 20) is what is actually enforced and
   already shown above the uploader. */
[data-testid="stFileUploaderDropzoneInstructions"] {display: none;}

/* Cards and status */
[data-testid="stStatusWidget"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
}
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    border-color: rgba(255, 255, 255, 0.09);
}
.stTabs [data-baseweb="tab"] {color: #9b9b9b;}
.stTabs [aria-selected="true"] {color: #f3f3f3 !important;}

/* Alerts / buttons / inputs / dataframes */
.stButton > button {
    border-radius: 10px;
    border-color: rgba(255, 255, 255, 0.14);
    color: #ececec;
}
.stButton > button[kind="primary"] {background: #10a37f; border-color: #10a37f;}
.stButton > button[kind="primary"]:hover {background: #0e8e6e; border-color: #0e8e6e;}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    color: #ececec !important;
}
[data-baseweb="popover"] [role="listbox"] {color: #ececec;}
[data-testid="stDataFrame"], [data-testid="styledDataGridContainer"] {
    color: #ececec;
}

/* Narrow windows: shrink horizontal paddings, never overlap */
@media (max-width: 900px) {
    .block-container {padding-left: 0.7rem; padding-right: 0.7rem;
        max-width: 100%;}
    .pkta-user-bubble {max-width: 92%;}
}
</style>
"""


def apply_theme() -> None:
    """Inject the application theme. Call once per page, after set_page_config."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
