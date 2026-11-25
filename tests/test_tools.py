"""C3: tools - search_documents, list_tasks, restricted create_task."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from core.models import TaskProposal, TaskStatus
from tools.document_tools import make_search_documents_tool
from tools.task_tools import make_create_task_tool, make_list_tasks_tool


@pytest.fixture
def seeded_tasks(task_repo, db):
    proposals = [
        TaskProposal(
            title="Finish ML assignment",
            description="Chapter 4 exercises",
            due_date=date.today() + timedelta(days=2),
            confirmation_message="ok",
        ),
        TaskProposal(title="Buy groceries", confirmation_message="ok"),
        TaskProposal(title="Old completed task", confirmation_message="ok"),
    ]
    for index, proposal in enumerate(proposals):
        task_repo.create_task(proposal, f"op-seed-{index}")
    # Simulate historical data: one task already completed. The application
    # UI itself cannot change task status (out of scope), so we set it
    # directly in the database.
    records = task_repo.list_tasks()
    target = next(r for r in records if r.title == "Old completed task")
    db.execute_write(
        "UPDATE tasks SET status = 'completed' WHERE id = ?", (target.id,)
    )
    return task_repo


# ---------------------------------------------------------------------------
# search_documents (read-only)
# ---------------------------------------------------------------------------
def test_search_documents_returns_passages_with_source_ids(ingestion, retriever):
    text = "Qdrant was selected as the vector database for the project."
    result = ingestion.ingest_file("plan.txt", text.encode())
    assert result.status == "indexed"

    collector: dict = {}
    tool = make_search_documents_tool(retriever, top_k=3, collector=collector)
    output = tool.invoke({"query": text})
    assert "Qdrant" in output
    assert "source_id=" in output
    assert len(collector) == 1
    chunk = next(iter(collector.values()))
    assert chunk["filename"] == "plan.txt"


def test_search_documents_empty_index(retriever):
    collector: dict = {}
    tool = make_search_documents_tool(retriever, top_k=3, collector=collector)
    output = tool.invoke({"query": "anything at all"})
    assert "No relevant passages" in output
    assert collector == {}


def test_search_documents_rejects_invalid_arguments(retriever):
    tool = make_search_documents_tool(retriever, top_k=3, collector={})
    with pytest.raises(ValidationError):
        tool.invoke({"query": "x"})  # too short (min_length=2)
    with pytest.raises(ValidationError):
        tool.invoke({})  # missing query


def test_search_documents_is_read_only(ingestion, retriever, doc_repo):
    ingestion.ingest_file("plan.txt", b"Some meaningful content about databases.")
    before = doc_repo.list_documents()
    make_search_documents_tool(retriever, top_k=3, collector={}).invoke(
        {"query": "Some meaningful content about databases."}
    )
    assert doc_repo.list_documents() == before


# ---------------------------------------------------------------------------
# list_tasks (read-only)
# ---------------------------------------------------------------------------
def test_list_tasks_returns_tasks(seeded_tasks):
    tool = make_list_tasks_tool(seeded_tasks)
    output = tool.invoke({"status": "all"})
    assert "Finish ML assignment" in output
    assert "Buy groceries" in output


def test_list_tasks_filters_by_status(seeded_tasks):
    tool = make_list_tasks_tool(seeded_tasks)
    output_open = tool.invoke({"status": "open"})
    assert "Finish ML assignment" in output_open
    assert "Old completed task" not in output_open
    output_completed = tool.invoke({"status": "completed"})
    assert "Old completed task" in output_completed
    assert "Finish ML assignment" not in output_completed


def test_list_tasks_empty_library(task_repo):
    tool = make_list_tasks_tool(task_repo)
    output = tool.invoke({"status": "all"})
    assert "no tasks" in output.lower()


def test_list_tasks_never_modifies_records(seeded_tasks):
    before = seeded_tasks.list_tasks()
    tool = make_list_tasks_tool(seeded_tasks)
    tool.invoke({"status": "all"})
    tool.invoke({"status": "open"})
    tool.invoke({"status": "completed"})
    after = seeded_tasks.list_tasks()
    assert before == after


def test_list_tasks_rejects_invalid_status(seeded_tasks):
    tool = make_list_tasks_tool(seeded_tasks)
    with pytest.raises(ValidationError):
        tool.invoke({"status": "deleted"})


# ---------------------------------------------------------------------------
# create_task (restricted write tool)
# ---------------------------------------------------------------------------
def test_create_task_with_valid_approved_input(task_repo):
    tool = make_create_task_tool(task_repo)
    raw = tool.invoke(
        {
            "title": "Review lecture notes",
            "description": "Chapter 4",
            "due_date": (date.today() + timedelta(days=1)).isoformat(),
            "priority": "high",
            "operation_id": "op-create-12345678",
        }
    )
    result = json.loads(raw)
    assert result["status"] == "created"
    assert result["task"]["title"] == "Review lecture notes"
    assert task_repo.count_tasks() == 1


def test_create_task_rejects_invalid_priority(task_repo):
    tool = make_create_task_tool(task_repo)
    with pytest.raises(ValidationError):
        tool.invoke(
            {
                "title": "Bad priority task",
                "priority": "urgent",
                "operation_id": "op-create-99999999",
            }
        )
    assert task_repo.count_tasks() == 0


def test_create_task_rejects_invalid_title(task_repo):
    tool = make_create_task_tool(task_repo)
    with pytest.raises(ValidationError):
        tool.invoke({"title": "no", "operation_id": "op-create-88888888"})
    assert task_repo.count_tasks() == 0


def test_create_task_rejects_invalid_date(task_repo):
    tool = make_create_task_tool(task_repo)
    raw = tool.invoke(
        {
            "title": "Invalid date task",
            "due_date": "not-a-date",
            "operation_id": "op-create-77777777",
        }
    )
    assert json.loads(raw)["status"] == "rejected"
    assert task_repo.count_tasks() == 0


def test_create_task_idempotent_on_operation_id(task_repo):
    tool = make_create_task_tool(task_repo)
    args = {
        "title": "Only once",
        "operation_id": "op-dedupe-12345678",
    }
    first = json.loads(tool.invoke(dict(args)))
    second = json.loads(tool.invoke(dict(args)))
    assert first["status"] == "created"
    assert second["status"] == "already_exists"
    assert first["task"]["id"] == second["task"]["id"]
    assert task_repo.count_tasks() == 1


def test_create_task_requires_operation_id(task_repo):
    tool = make_create_task_tool(task_repo)
    with pytest.raises(ValidationError):
        tool.invoke({"title": "No operation id"})
