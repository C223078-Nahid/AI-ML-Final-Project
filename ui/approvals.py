"""Approval Center: pending task proposals awaiting human decision.

Nothing here writes tasks directly - approval resumes the interrupted
LangGraph workflow, which is the only path to the create_task tool.
"""

from __future__ import annotations

import streamlit as st

from ui.graph_service import get_thread_state, resume_operation


def render_proposal_card(
    container,
    operation_id: str,
    proposal: dict,
    thread_id: str,
    show_context: bool = True,
) -> None:
    """Render one proposal with explicit Approve / Reject actions."""
    with st.container(border=True):
        st.markdown(f"**Proposed task:** {proposal.get('title', '(untitled)')}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"Priority: **{proposal.get('priority', 'medium')}**")
        with col2:
            st.markdown(f"Due date: **{proposal.get('due_date') or 'none'}**")
        with col3:
            st.markdown(f"Operation ID: `{operation_id[:13]}...`")
        if proposal.get("description"):
            st.markdown(f"**Description:** {proposal['description']}")
        if proposal.get("confirmation_message"):
            st.caption(proposal["confirmation_message"])

        if show_context:
            snapshot = get_thread_state(container, thread_id)
            if snapshot:
                messages = snapshot.values.get("messages") or []
                recent = [
                    str(m.content)
                    for m in messages
                    if m.__class__.__name__ == "HumanMessage"
                ]
                if recent:
                    with st.expander("Conversation context"):
                        st.write(f"User: {recent[-1]}")

        st.warning(
            "This task has NOT been saved yet. It is only created after you "
            "approve it."
        )
        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button(
                "Approve", key=f"approve-{operation_id}", type="primary"
            ):
                with st.spinner("Saving task..."):
                    resume_operation(container, thread_id, operation_id, "approved")
                st.rerun()
        with col_reject:
            if st.button("Reject", key=f"reject-{operation_id}"):
                resume_operation(container, thread_id, operation_id, "rejected")
                st.rerun()


def render(container) -> None:
    """Render the Approval Center page."""
    # Page title/subtitle are rendered by the application header (app.py).
    pending = container.conversation_repo.list_pending_operations("pending")
    if not pending:
        st.success("No pending task proposals.")
        return

    st.info(f"{len(pending)} proposal(s) waiting for your decision.")
    for operation in pending:
        render_proposal_card(
            container,
            operation_id=operation.operation_id,
            proposal=operation.proposal,
            thread_id=operation.thread_id,
            show_context=True,
        )
