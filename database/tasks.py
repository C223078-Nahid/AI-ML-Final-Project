"""Task repository: the only component allowed to write the ``tasks`` table.

``create_task`` is idempotent on ``operation_id``: approving the same task
proposal twice (double click, browser refresh, resumed graph) still results in
exactly one row.
"""

from __future__ import annotations

import sqlite3

from core.models import (
    TaskCreationResult,
    TaskPriority,
    TaskProposal,
    TaskRecord,
    TaskStatus,
    new_id,
    utc_now_iso,
)
from database.connection import Database


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- reads ------------------------------------------------------------------
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            id=row["id"],
            operation_id=row["operation_id"],
            title=row["title"],
            description=row["description"] or "",
            due_date=row["due_date"],
            priority=TaskPriority(row["priority"]),
            status=TaskStatus(row["status"]),
            created_at=row["created_at"],
        )

    def list_tasks(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        """List tasks, newest first. Read-only: never modifies records."""
        if status is None:
            rows = self._db.query_all(
                "SELECT * FROM tasks ORDER BY created_at DESC, id"
            )
        else:
            rows = self._db.query_all(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC, id",
                (status.value,),
            )
        return [self._row_to_record(row) for row in rows]

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self._db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return self._row_to_record(row) if row else None

    def get_by_operation_id(self, operation_id: str) -> TaskRecord | None:
        row = self._db.query_one(
            "SELECT * FROM tasks WHERE operation_id = ?", (operation_id,)
        )
        return self._row_to_record(row) if row else None

    def count_tasks(self, status: TaskStatus | None = None) -> int:
        if status is None:
            row = self._db.query_one("SELECT COUNT(*) AS n FROM tasks")
        else:
            row = self._db.query_one(
                "SELECT COUNT(*) AS n FROM tasks WHERE status = ?", (status.value,)
            )
        return int(row["n"]) if row else 0

    # -- writes -------------------------------------------------------------------
    def create_task(
        self, proposal: TaskProposal, operation_id: str
    ) -> TaskCreationResult:
        """Create a task from an approved proposal.

        Idempotent: if a task with the same ``operation_id`` already exists it
        is returned unchanged with ``created=False``.
        """
        existing = self.get_by_operation_id(operation_id)
        if existing is not None:
            return TaskCreationResult(task=existing, created=False)

        record = TaskRecord(
            id=new_id(),
            operation_id=operation_id,
            title=proposal.title,
            description=proposal.description,
            due_date=proposal.due_date.isoformat() if proposal.due_date else None,
            priority=proposal.priority,
            status=TaskStatus.OPEN,
            created_at=utc_now_iso(),
        )
        try:
            self._db.execute_write(
                """
                INSERT INTO tasks
                    (id, operation_id, title, description, due_date,
                     priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.operation_id,
                    record.title,
                    record.description,
                    record.due_date,
                    record.priority.value,
                    record.status.value,
                    record.created_at,
                ),
            )
        except sqlite3.IntegrityError:
            # A concurrent write with the same operation_id won the race.
            existing = self.get_by_operation_id(operation_id)
            if existing is not None:
                return TaskCreationResult(task=existing, created=False)
            raise
        return TaskCreationResult(task=record, created=True)
