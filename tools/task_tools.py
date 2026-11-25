"""Task tools (concept C3, tools 2 and 3).

- ``list_tasks`` is read-only and safe to bind to a model.
- ``create_task`` performs the only database write. It is NEVER bound to a
  model or exposed through an agent ToolNode: the graph's task_execution
  node invokes it directly with an application-generated operation_id, and
  only after the human_approval node resumed with an explicit approval.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from core.models import TaskProposal, TaskPriority, TaskStatus
from database.tasks import TaskRepository


class ListTasksInput(BaseModel):
    status: Literal["open", "completed", "cancelled", "all"] = Field(
        default="all", description="Filter tasks by status, or 'all'."
    )


def format_task_line(record) -> str:
    due = f", due {record.due_date}" if record.due_date else ""
    return (
        f"- [{record.status.value}] {record.title} "
        f"({record.priority.value} priority{due}) — id: {record.id[:8]}"
    )


def make_list_tasks_tool(task_repo: TaskRepository) -> Any:
    @tool("list_tasks", args_schema=ListTasksInput)
    def list_tasks(status: str = "all") -> str:
        """List the user's tasks, optionally filtered by status.

        Use this when the user asks about their tasks, deadlines, or what
        they have to do. Each task shows its status, title, priority, due
        date and a short id that can be referenced in conversation.
        """
        status_filter = None if status == "all" else TaskStatus(status)
        records = task_repo.list_tasks(status_filter)
        if not records:
            if status_filter is None:
                return "You have no tasks yet. Ask me to create one and I will propose it for your approval."
            return f"You have no {status} tasks."
        lines = "\n".join(format_task_line(record) for record in records)
        return f"You have {len(records)} task(s):\n{lines}"

    return list_tasks


class CreateTaskInput(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=2000)
    due_date: str | None = Field(
        default=None, description="ISO date (YYYY-MM-DD) or null."
    )
    priority: Literal["low", "medium", "high"] = "medium"
    operation_id: str = Field(
        min_length=8,
        description="Application-generated unique operation id used for idempotency.",
    )


def make_create_task_tool(task_repo: TaskRepository) -> Any:
    @tool("create_task", args_schema=CreateTaskInput)
    def create_task(
        title: str,
        description: str = "",
        due_date: str | None = None,
        priority: str = "medium",
        operation_id: str = "",
    ) -> str:
        """RESTRICTED WRITE TOOL: create a task.

        This tool may only be invoked by the application's task_execution
        graph node after the user explicitly approved the proposal. It is
        never bound to a language model.
        """
        try:
            parsed_due = date.fromisoformat(due_date) if due_date else None
            proposal = TaskProposal(
                title=title,
                description=description,
                due_date=parsed_due,
                priority=TaskPriority(priority),
                confirmation_message="(approved by user)",
            )
        except ValueError as exc:
            return json.dumps({"status": "rejected", "error": str(exc)})
        result = task_repo.create_task(proposal, operation_id)
        return json.dumps(
            {
                "status": "created" if result.created else "already_exists",
                "task": result.task.model_dump(mode="json"),
            }
        )

    return create_task
