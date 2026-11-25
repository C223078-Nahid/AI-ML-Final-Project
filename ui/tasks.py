"""Task Management page: listing and filtering (creation happens via chat)."""

from __future__ import annotations

import streamlit as st

from core.models import TaskStatus


def render(container) -> None:
    # Page title/subtitle are rendered by the application header (app.py).
    filter_options = ["All", "Open", "Completed", "Cancelled"]
    choice = st.selectbox("Filter by status", filter_options, index=0)
    status = {
        "All": None,
        "Open": TaskStatus.OPEN,
        "Completed": TaskStatus.COMPLETED,
        "Cancelled": TaskStatus.CANCELLED,
    }[choice]

    records = container.task_repo.list_tasks(status)
    if not records:
        st.info(
            "No tasks match this filter. Ask the assistant in the Chat page "
            "to propose a new task."
        )
        return

    st.caption(
        f"{len(records)} task(s)"
        + (f" with status '{status.value}'" if status else "")
    )
    rows = [
        {
            "Title": record.title,
            "Description": record.description or "—",
            "Due date": record.due_date or "—",
            "Priority": record.priority.value,
            "Status": record.status.value,
            "Created": record.created_at[:16].replace("T", " "),
        }
        for record in records
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
