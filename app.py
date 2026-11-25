"""Personal Knowledge & Task Assistant - Streamlit entry point.

Run with:  streamlit run app.py

Note: .env is loaded BEFORE any LangChain/LangSmith import so that
LANGSMITH_* tracing variables are honored from the very first call.

Layout (follow-up task 3, ChatGPT-style sidebar):
- Sidebar, left-aligned, in this exact hierarchy: application name/logo at
  the very top-left, New Chat, then application navigation (Documents,
  Tasks, Settings, Approvals), then the Chats history section (visually
  separated, independently scrollable).  No taglines, no footer text.
- Main area: a slim page-title header followed by the active page.  The
  sidebar toggle sits at the top-right OF THE SIDEBAR row (next to the
  brand); when the sidebar is collapsed, a reopen control floats at the
  left edge of the application - the same top-left screen corner in both
  states.

Sidebar collapse/restore (preserved from the previous fix):
- Visibility state is ``pkta_sidebar_open`` in session_state: it survives
  ordinary reruns, does not touch the active conversation
  (``pkta_thread``/``pkta_page``), and cannot interrupt pending LangGraph
  workflows or human-approval interrupts (all of those live in the SQLite
  checkpointer, not in UI state).
- The application state is authoritative for sidebar visibility; the
  native Streamlit collapse/expand controls stay hidden so there is exactly
  one source of truth.

Session-state contract (temporary UI state only - all application data is
persisted in SQLite/Qdrant/checkpoints):
- ``pkta_page``       active page id (chat / documents / tasks / approvals / settings)
- ``pkta_thread``     active conversation thread id (None = fresh chat)
- ``pkta_sidebar_open``   sidebar visibility (True by default)
- ``pkta_confirm_delete``  thread id awaiting delete confirmation
- ``pkta_processed_uploads``  upload signatures already ingested (rerun guard)
- ``pkta_pending_attachments``  ingested documents staged as composer chips
- ``pkta_processing``   active generation job token (background worker)
- ``pkta_queued``       message waiting for a winding-down generation to exit
- ``pkta_last_request``  dedup fingerprint of the most recent executed request
"""

from dotenv import load_dotenv

load_dotenv()

import threading  # noqa: E402

import streamlit as st  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from config.settings import get_settings  # noqa: E402
from core.llm_gateway import ModelGateway  # noqa: E402
from core.web_search import build_search_service  # noqa: E402
from database.connection import Database  # noqa: E402
from database.conversations import ConversationRepository  # noqa: E402
from database.documents import DocumentRepository  # noqa: E402
from database.memory import MemoryRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from database.tasks import TaskRepository  # noqa: E402
from graph.nodes import GraphServices  # noqa: E402
from graph.persistence import create_checkpointer  # noqa: E402
from graph.workflow import build_workflow  # noqa: E402
from rag.embeddings import LocalEmbeddings  # noqa: E402
from rag import diagnostics  # noqa: E402
from rag.ingestion import IngestionService  # noqa: E402
from rag.lexical_index import LexicalIndex  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402
from tools.registry import build_default_registry  # noqa: E402
from ui.theme import apply_theme  # noqa: E402

PAGES = ("chat", "documents", "tasks", "approvals", "settings")

# Application-header copy per page (titles/subtitles previously rendered by
# each page itself; centralized here so the sidebar toggle sits next to a
# single consistent header).
PAGE_META = {
    "chat": ("Chat", "Ask anything — documents, tasks, or general chat."),
    "documents": (
        "Document Library",
        "Shared across every chat — upload once, ask anywhere.",
    ),
    "tasks": (
        "Tasks",
        "Tasks are created through the chat assistant after your explicit "
        "approval. This page lists and filters them.",
    ),
    "approvals": (
        "Approval Center",
        "Task creation always requires your explicit approval. Proposals "
        "survive page refreshes and application restarts until you decide.",
    ),
    "settings": (
        "Settings",
        "Preferences, model status, and local storage.",
    ),
}

#: Sidebar visibility is OWNED by the application (``pkta_sidebar_open``),
#: not by Streamlit's native collapse state (which persists in the browser's
#: localStorage and can desync - e.g. a user who collapsed the sidebar under
#: the previous build). Exactly one of the two blocks below is injected on
#: every rerun, so the visible state always matches the app state:
#: - OPEN: force the native sidebar to the expanded geometry even if the
#:   browser's persisted native state is "collapsed".
#: - COLLAPSED: hide the sidebar entirely; the main area (a flex sibling)
#: expands to the full width automatically.
#: The native expand control is hidden in both states: the app-header toggle
#: is the single, always-visible control (the reported bug was that NO
#: visible control existed to restore the sidebar).
SIDEBAR_OPEN_CSS = (
    "<style>"
    '[data-testid="stSidebar"]{width:300px !important;'
    "min-width:300px !important;max-width:300px !important;"
    'transform:none !important;}'
    '[data-testid="stExpandSidebarButton"]{display:none !important;}'
    "</style>"
)
SIDEBAR_COLLAPSED_CSS = (
    "<style>"
    '[data-testid="stSidebar"]{display:none !important;}'
    '[data-testid="stExpandSidebarButton"]{display:none !important;}'
    "</style>"
)


@dataclass
class AppContainer:
    """Bundle of every long-lived service used by the UI."""

    settings: object
    db: Database
    doc_repo: DocumentRepository
    task_repo: TaskRepository
    memory_repo: MemoryRepository
    conversation_repo: ConversationRepository
    store: VectorStore
    lexical: LexicalIndex
    retriever: Retriever
    gateway: ModelGateway
    ingestion: IngestionService
    graph: object
    checkpointer: object
    # Centralized tool registry + web search service (follow-up task 4).
    tool_registry: object = None
    web_search: object = None


def build_container(settings) -> AppContainer:
    """Wire every component together (pure function, no Streamlit cache)."""
    settings.ensure_directories()
    db = Database(settings.sqlite_path)
    init_schema(db)

    doc_repo = DocumentRepository(db)
    task_repo = TaskRepository(db)
    memory_repo = MemoryRepository(db)
    conversation_repo = ConversationRepository(db)

    embeddings = LocalEmbeddings(settings.embedding_model)
    store = VectorStore(settings.qdrant_path, embeddings)
    # Hybrid retrieval: Qdrant semantics + SQLite FTS5 keywords (follow-up
    # task 3.4). Both halves share the same ingestion lifecycle, so their
    # contents stay consistent.
    lexical = LexicalIndex(db)
    retriever = Retriever(
        store,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        lexical_index=lexical,
    )
    gateway = ModelGateway.from_settings(settings)
    ingestion = IngestionService(store, doc_repo, settings, lexical_index=lexical)

    # Startup reconciliation (follow-up task 3.1/3.3): a previous crash or
    # a manual data reset can leave orphan Qdrant points (the reported
    # "retrieval returned 2 chunks while the UI said 1" inconsistency).
    # Removing them here guarantees Qdrant, the SQLite bookkeeping, and the
    # UI chunk counts agree for the whole session.
    try:
        valid_ids = {
            record.document_id
            for record in doc_repo.list_documents()
            if record.status == "indexed"
        }
        report = store.reconcile(valid_ids)
        report["fts_rows_removed"] = lexical.reconcile()
        diagnostics.log_reconciliation(report)
    except Exception as exc:  # noqa: BLE001 - never block startup
        diagnostics.log_retrieval_failure("startup_reconciliation", exc)

    services = GraphServices(
        gateway=gateway,
        retriever=retriever,
        task_repo=task_repo,
        memory_repo=memory_repo,
        conversation_repo=conversation_repo,
        settings=settings,
        doc_repo=doc_repo,
        tool_registry=None,   # set below (needs the search service)
        web_search=None,
    )
    # Provider-independent web search service (follow-up task 4, section
    # 9) and the centralized tool registry (section 6.2): one shared
    # instance so the agent tools, the fact-check workflow, and the
    # Settings page all see the same availability and URL allowlist.
    search_service = build_search_service(settings)
    tool_registry = build_default_registry(
        retriever=retriever,
        doc_repo=doc_repo,
        task_repo=task_repo,
        settings=settings,
        search_service=search_service,
    )
    services.tool_registry = tool_registry
    services.web_search = search_service
    checkpointer = create_checkpointer(settings.checkpoint_db_path)
    graph = build_workflow(services, checkpointer)

    container = AppContainer(
        settings=settings,
        db=db,
        doc_repo=doc_repo,
        task_repo=task_repo,
        memory_repo=memory_repo,
        conversation_repo=conversation_repo,
        store=store,
        lexical=lexical,
        retriever=retriever,
        gateway=gateway,
        ingestion=ingestion,
        graph=graph,
        checkpointer=checkpointer,
        tool_registry=tool_registry,
        web_search=search_service,
    )
    # Pre-load the embedding model in the background so the FIRST upload of
    # a session does not pay the model-load cost (a major cause of the
    # previously reported slow first ingestion).
    if isinstance(embeddings, LocalEmbeddings):

        def _warm() -> None:
            try:
                embeddings.warmup()
            except Exception:  # noqa: BLE001 - warmup is best effort
                pass

        threading.Thread(target=_warm, daemon=True).start()
    return container


@st.cache_resource(show_spinner=False)
def get_container() -> AppContainer:
    return build_container(get_settings())


# ---------------------------------------------------------------------------
# Sidebar (follow-up task, section 3: exact hierarchy, left-aligned)
# ---------------------------------------------------------------------------
def _brand() -> None:
    """The application name - and nothing else.

    The former 'local · private · yours' sub-line and the footer tagline
    ("RAG over your documents ...") were removed per the task; no
    replacement promotional text is shown anywhere.
    """
    st.markdown(
        '<div class="pkta-brand">'
        '<div class="pkta-logo">KT</div>'
        '<div class="pkta-name">Knowledge &amp; Tasks</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar(container) -> None:
    """Sidebar in the exact order required by the task:

    Top section:  1. application name/logo (top-left)   2. New Chat button
    Navigation:   3. Documents   4. Tasks   5. Settings
                  (+ Approvals, the remaining existing page, kept with the
                   navigation group after Settings so D→T→S order holds)
    Chat history: 6. Chats heading   7. previous conversations

    Everything is left-aligned; the chat history is visually separated
    from the application navigation and scrolls independently.
    """
    # -- 1. brand at the top-left + collapse toggle on the same row ----------
    col_brand, col_toggle = st.columns(
        [5, 0.34], gap="small", vertical_alignment="center"
    )
    with col_brand:
        _brand()
    with col_toggle:
        if st.button(
            "☰",
            key="pkta_sidebar_collapse",
            help="Close sidebar",
            use_container_width=True,
        ):
            st.session_state["pkta_sidebar_open"] = False
            st.rerun()

    # -- 2. New chat -----------------------------------------------------------
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        st.session_state["pkta_thread"] = None
        st.session_state["pkta_page"] = "chat"
        st.rerun()

    # -- 3.-5. application navigation (Documents, Tasks, Settings, Approvals) -
    nav_items = [
        ("documents", "📄  Documents"),
        ("tasks", "✅  Tasks"),
        ("settings", "⚙️  Settings"),
        ("approvals", "⏳  Approvals"),
    ]
    for page_id, page_label in nav_items:
        count_badge = ""
        if page_id == "documents":
            n_docs = container.doc_repo.count_documents(("indexed",))
            count_badge = f"  ({n_docs})"
        elif page_id == "approvals":
            n_pending = len(
                container.conversation_repo.list_pending_operations("pending")
            )
            count_badge = f"  ({n_pending})" if n_pending else ""
        if st.button(
            f"{page_label}{count_badge}",
            key=f"nav-{page_id}",
            use_container_width=True,
        ):
            st.session_state["pkta_page"] = page_id
            st.rerun()

    # -- 6./7. chat history (separated, independently scrollable) --------------
    with st.container():
        st.markdown('<div class="pkta-history-scroll-start"></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="pkta-history-title">Chats</div>',
                    unsafe_allow_html=True)

        conversations = container.conversation_repo.list_conversations()
        active_thread = st.session_state.get("pkta_thread")

        pending_delete = st.session_state.get("pkta_confirm_delete")
        if pending_delete:
            record = container.conversation_repo.get_conversation(pending_delete)
            name = (record.title if record and record.title
                    else "this chat") or "this chat"
            st.warning(f"Delete **{name}**? The shared document library "
                       "and tasks are kept.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Delete chat", use_container_width=True):
                    container.conversation_repo.delete_conversation(
                        pending_delete
                    )
                    if st.session_state.get("pkta_thread") == pending_delete:
                        st.session_state["pkta_thread"] = None
                        st.session_state["pkta_page"] = "chat"
                    st.session_state["pkta_confirm_delete"] = None
                    st.rerun()
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["pkta_confirm_delete"] = None
                    st.rerun()

        if not conversations and not pending_delete:
            st.caption("No conversations yet. Start a new chat!")

        for conv in conversations:
            is_active = conv.thread_id == active_thread
            col_chat, col_del = st.columns([5, 0.8])
            with col_chat:
                label = conv.title or "Untitled chat"
                if st.button(
                    label,
                    key=f"chat-{conv.thread_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["pkta_thread"] = conv.thread_id
                    st.session_state["pkta_page"] = "chat"
                    st.rerun()
            with col_del:
                if st.button("×", key=f"delchat-{conv.thread_id}",
                             help="Delete this conversation"):
                    st.session_state["pkta_confirm_delete"] = conv.thread_id
                    st.rerun()


# ---------------------------------------------------------------------------
# Page header (title only) + collapsed-sidebar reopen rail
# ---------------------------------------------------------------------------
def _chat_title(container) -> str:
    """Title text for the chat page: the active conversation, if any."""
    thread_id = st.session_state.get("pkta_thread")
    if not thread_id:
        return "New chat"
    record = container.conversation_repo.get_conversation(thread_id)
    return (record.title if record and record.title else None) or "Chat"


def _render_sidebar_rail() -> None:
    """Reopen control shown at the LEFT EDGE while the sidebar is collapsed.

    ChatGPT-style: a single floating toggle at the top-left corner of the
    application. It lives in the main area (the sidebar itself is hidden),
    positioned fixed at the left edge via the ``pkta-rail-marker`` selector,
    and it only flips ``pkta_sidebar_open`` - never the conversation state.
    """
    col_rail, _ = st.columns([0.075, 5])
    with col_rail:
        st.markdown('<div class="pkta-rail-marker"></div>',
                    unsafe_allow_html=True)
        if st.button(
            "☰",
            key="pkta_sidebar_reopen",
            help="Open sidebar",
            use_container_width=True,
        ):
            st.session_state["pkta_sidebar_open"] = True
            st.rerun()


def _render_app_header(container, page: str) -> None:
    """Slim page-title header (the toggle lives on the sidebar side)."""
    if page == "chat":
        title, subtitle = _chat_title(container), PAGE_META["chat"][1]
    else:
        title, subtitle = PAGE_META.get(page, ("Chat", ""))
    sub_html = (
        f'<div class="pkta-page-sub">{_escape(subtitle)}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f'<div class="pkta-page-header">{_escape(title)}{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _escape(text: str) -> str:
    import html

    return html.escape(str(text or ""))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def _render_page(container, page: str) -> None:
    if page == "chat":
        from ui.chat import render

        render(container)
    elif page == "documents":
        from ui.documents import render

        render(container)
    elif page == "tasks":
        from ui.tasks import render

        render(container)
    elif page == "approvals":
        from ui.approvals import render

        render(container)
    elif page == "settings":
        from ui.settings import render

        render(container)


def main() -> None:
    st.set_page_config(
        page_title="Personal Knowledge & Task Assistant",
        layout="centered",
        initial_sidebar_state="auto",
    )
    apply_theme()
    # Safe RAG diagnostics ([rag-diag] lines: collection, point counts,
    # retrieval execution, context inclusion - never document content).
    from rag.diagnostics import attach_stderr_handler

    attach_stderr_handler()

    settings = get_settings()
    errors, warnings = settings.validate()
    container = get_container()

    if "pkta_page" not in st.session_state:
        st.session_state["pkta_page"] = "chat"
    if "pkta_sidebar_open" not in st.session_state:
        st.session_state["pkta_sidebar_open"] = True

    # The application state is authoritative for sidebar visibility; the
    # matching CSS block is injected once per rerun, before any content.
    sidebar_open = st.session_state.get("pkta_sidebar_open", True)
    st.markdown(
        SIDEBAR_OPEN_CSS if sidebar_open else SIDEBAR_COLLAPSED_CSS,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        _render_sidebar(container)

    if errors:
        for error in errors:
            st.error(error)
        st.stop()
    for warning in warnings:
        st.warning(warning)

    page = st.session_state["pkta_page"]
    if page not in PAGES:
        page = "chat"

    # Collapsed sidebar: the reopen control floats at the left edge of the
    # application (top-left corner - the same position the toggle occupies
    # inside the sidebar when it is open).
    if not sidebar_open:
        _render_sidebar_rail()

    _render_app_header(container, page)
    _render_page(container, page)


if __name__ == "__main__":
    main()
