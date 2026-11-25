"""Settings page: long-term preferences, model status, storage info.

Never displays API keys, environment values, or internal prompts.

Truncation fix (root cause of labels like "return" showing as "retur"):
- No fixed-width text containers; every column can shrink (``min-width: 0``
  is enforced globally by the theme CSS on flex children).
- ``overflow-wrap: anywhere`` on all text so long identifiers (model names,
  file paths) wrap instead of being clipped by a neighbouring control.
- Dropdown/option lists keep their native width; nothing overlaps them.
"""

from __future__ import annotations

import streamlit as st

from database.memory import PreferenceError

EXAMPLE_KEYS = [
    "preferred_response_style",
    "default_task_priority",
    "preferred_task_format",
    "explanation_level",
]


def _render_preferences(container) -> None:
    st.subheader("Long-term preferences")

    with st.form("save_preference", border=True):
        st.caption(
            "Preferences are stored locally and injected into every new "
            "conversation, in every chat. They are only saved when you "
            "explicitly submit this form - nothing is memorized "
            "automatically. Secrets are refused."
        )
        key = st.text_input("Key", placeholder="e.g. preferred_response_style")
        value = st.text_area(
            "Value",
            placeholder="e.g. Keep answers short and use bullet points.",
            height=90,
        )
        submitted = st.form_submit_button("Save preference", type="primary")
        if submitted:
            try:
                saved = container.memory_repo.save_preference(key, value)
                st.success(f"Saved preference '{saved.key}'.")
            except PreferenceError as exc:
                st.error(str(exc))

    preferences = container.memory_repo.list_preferences()
    if not preferences:
        st.info(
            "No preferences saved yet. Example keys: " + ", ".join(EXAMPLE_KEYS)
        )
        return

    for pref in preferences:
        col_key, col_value, col_del = st.columns([2.2, 4.6, 0.9])
        with col_key:
            st.markdown(f"**{pref.key}**")
        with col_value:
            st.write(pref.value)
        with col_del:
            if st.button("Delete", key=f"delpref-{pref.id}",
                         help=f"Delete preference {pref.key}"):
                container.memory_repo.delete_preference(pref.id)
                st.rerun()
        st.caption(
            f"updated {pref.updated_at[:16].replace('T', ' ')}"
        )


def _render_status(container) -> None:
    st.subheader("Runtime status")
    summary = container.settings.public_summary()
    gateway_info = container.gateway.describe()

    st.markdown("**Language models**")
    provider = summary["llm_primary_provider"]
    primary = summary["primary_model"]
    st.markdown(
        f"- Active primary provider: `{provider}` — model `{primary}` — "
        + (
            "available"
            if gateway_info["primary_available"]
            else "disabled (missing credentials)"
        )
    )
    st.markdown("- Fallback chain (in failover order):")
    for entry in gateway_info["chain"][1:]:
        state = (
            "available"
            if entry["available"]
            else "skipped (not configured)"
        )
        st.markdown(
            f"  - `{entry['provider']}` — model `{entry['model']}` — {state}"
        )
    if summary.get("cloudflare_enabled") and not summary.get(
        "cloudflare_available"
    ):
        st.caption(
            "Cloudflare GLM-4.7-Flash is enabled in the chain but not fully "
            "configured: set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "in .env (see the README section on Cloudflare credentials)."
        )
    st.caption(
        "When a provider fails with a rate limit, quota exhaustion, "
        "timeout, or transient error, the next provider in the chain is used "
        "automatically - the request, its conversation context, and any "
        "retrieved document passages are preserved. Configure the order "
        "with LLM_PRIMARY_PROVIDER and LLM_FALLBACK_PROVIDERS in .env."
    )

    st.markdown("**Observability**")
    tracing = summary["langsmith_tracing"]
    st.markdown(
        f"- LangSmith tracing: **{'enabled' if tracing else 'disabled'}**"
        + (f" — project `{summary['langsmith_project']}`" if tracing else "")
    )
    if not tracing:
        st.caption(
            "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to "
            "enable tracing."
        )

    st.markdown("**Local storage**")
    st.markdown(f"- Embedding model: `{summary['embedding_model']}` (runs locally)")
    st.markdown(f"- Application database: `{summary['sqlite_path']}`")
    st.markdown(f"- Checkpoint database: `{summary['checkpoint_db_path']}`")
    st.markdown(f"- Vector index (shared by all chats): `{summary['qdrant_path']}`")
    try:
        index_stats = container.store.stats()
        st.markdown(
            f"- Shared collection: `{index_stats['collection']}` — "
            f"**{index_stats['points']} indexed point(s)**"
            + (
                f", embedding dimension {index_stats['dimension']}"
                if index_stats.get("dimension")
                else ""
            )
        )
    except Exception:  # noqa: BLE001 - status display only
        pass
    st.caption(
        f"{container.doc_repo.count_documents()} indexed document(s) · "
        f"{container.task_repo.count_tasks()} task(s) · "
        f"{len(container.memory_repo.list_preferences())} preference(s)"
    )

    st.markdown("**Shared document library**")
    st.caption(
        "Every chat session searches the same document index. Conversation "
        "histories stay isolated per chat; uploaded documents are available "
        "everywhere until removed."
    )

    # --- Tool registry (follow-up task 4, section 6.2) --------------------
    registry = getattr(container, "tool_registry", None)
    if registry is not None:
        st.markdown("**Tool registry** (model-driven function calling)")
        st.caption(
            "The model decides when to call a READ-ONLY tool; every "
            "argument is validated against the tool's schema before "
            "execution. RESTRICTED tools are never offered to the model - "
            "task creation stays behind your explicit approval."
        )
        for entry in registry.describe():
            state = (
                "offered to the model"
                if entry["bindable"]
                else (
                    "unavailable (" + entry["unavailable_reason"] + ")"
                    if not entry["available"]
                    else "application-invoked only (restricted)"
                )
            )
            st.markdown(
                f"- `{entry['name']}` — {entry['permission']} — {state}"
            )
        summary = container.settings.public_summary()
        if not summary.get("llm_tool_calling_enabled", True):
            st.caption(
                "LLM_TOOL_CALLING_ENABLED is false: no tools are bound and "
                "the model answers directly."
            )

    # --- Web search (follow-up task 4, section 9) --------------------------
    st.markdown("**Web search**")
    web_summary = container.settings.public_summary()
    if web_summary.get("web_search_available"):
        st.markdown(
            f"- Provider: `{web_summary['web_search_provider']}` — "
            "available (the model can call web_search for independent "
            "evidence)"
        )
    elif web_summary.get("web_search_enabled"):
        st.markdown(
            f"- Provider: `{web_summary['web_search_provider']}` — "
            "unavailable: set TAVILY_API_KEY in .env to offer the "
            "web_search tool (get a free key at https://app.tavily.com)."
        )
    else:
        st.markdown("- Disabled (WEB_SEARCH_ENABLED=false).")
    st.caption(
        "Search results are treated as untrusted external data: they are "
        "cited as evidence but can never act as instructions, and page "
        "reading is restricted to URLs returned by web_search in the same "
        "request."
    )


def render(container) -> None:
    # Page title/subtitle are rendered by the application header (app.py).
    _render_preferences(container)
    st.divider()
    _render_status(container)
