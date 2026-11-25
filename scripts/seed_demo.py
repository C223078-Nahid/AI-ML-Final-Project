"""Seed a few demo tasks for testing and demonstrations.

Usage:  python scripts/seed_demo.py

Idempotent: running it twice does not create duplicates (stable operation
ids).  Tasks are written directly through the repository - the same code
path the approved workflow uses - never through the language model.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import get_settings
from core.models import TaskProposal, TaskStatus
from database.connection import Database
from database.schema import init_schema
from database.tasks import TaskRepository

DEMO_TASKS = [
    (
        "op-demo-finish-ml-assignment",
        TaskProposal(
            title="Finish Machine Learning assignment",
            description="Chapter 4 exercises on backpropagation",
            due_date=date.today() + timedelta(days=2),
            confirmation_message="Seed demo task.",
        ),
    ),
    (
        "op-demo-database-midterm",
        TaskProposal(
            title="Prepare for Database Systems midterm",
            description="Review normalization and indexing notes",
            due_date=date.today() + timedelta(days=6),
            confirmation_message="Seed demo task.",
        ),
    ),
    (
        "op-demo-project-documentation",
        TaskProposal(
            title="Write project README",
            description="Document installation and demo steps",
            due_date=date.today() + timedelta(days=10),
            confirmation_message="Seed demo task.",
        ),
    ),
]


def main() -> None:
    settings = get_settings()
    settings.ensure_directories()
    db = Database(settings.sqlite_path)
    init_schema(db)
    repo = TaskRepository(db)

    created = 0
    for operation_id, proposal in DEMO_TASKS:
        result = repo.create_task(proposal, operation_id)
        if result.created:
            created += 1
            print(f"created: {result.task.title}")
        else:
            print(f"already exists: {result.task.title}")

    # Mark the first demo task completed for filter demonstrations.
    first = repo.get_by_operation_id("op-demo-finish-ml-assignment")
    if first is not None and first.status is TaskStatus.OPEN:
        db.execute_write(
            "UPDATE tasks SET status = 'completed' WHERE id = ?", (first.id,)
        )
        print("marked completed: Finish Machine Learning assignment")

    print(f"\nDone. {repo.count_tasks()} task(s) in the database.")


if __name__ == "__main__":
    main()
