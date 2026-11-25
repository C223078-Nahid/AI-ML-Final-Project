"""Conversation and pending-operation bookkeeping.

``conversations`` is a registry of chat threads so the sidebar can list them
without scanning the LangGraph checkpoint database.  ``pending_operations``
indexes task proposals waiting for human approval so the Approval Center can
find them across threads, refreshes, and restarts.

The LangGraph checkpointer remains the source of truth for workflow state;
these tables are UI-facing indexes only.
"""

from __future__ import annotations

import json

from core.models import (
    ConversationInfo,
    PendingOperation,
    utc_now_iso,
)
from database.connection import Database


def title_from_message(text: str, max_chars: int = 48) -> str | None:
    """Derive a short conversation title from the first user message.

    Simple, deterministic, and LLM-free by design: collapse whitespace,
    trim to the last whole word within the limit, and fall back to None for
    blank input so the UI can show a default label instead.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0]
    return (cut + "…") if cut else cleaned[:max_chars] + "…"


class ConversationRepository:
    """Chat-session bookkeeping.

    Titles are derived from the FIRST user message of a thread and are never
    silently overwritten by later messages (see ``ensure_conversation``);
    ``set_title`` is the only explicit rename path.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- conversations ----------------------------------------------------------
    def ensure_conversation(self, thread_id: str, title: str | None = None) -> None:
        """Register a thread. The title is set once, on first registration.

        Repeated calls with different titles (e.g. every message of a chat
        passing its own text) must NOT rename the conversation: the sidebar
        shows the title derived from the first user message.
        """
        now = utc_now_iso()
        existing = self._db.query_one(
            "SELECT thread_id FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        if existing is None:
            self._db.execute_write(
                "INSERT INTO conversations (thread_id, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (thread_id, title, now, now),
            )

    def set_title(self, thread_id: str, title: str) -> None:
        cleaned = title.strip()
        if not cleaned:
            return
        now = utc_now_iso()
        self._db.execute_write(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE thread_id = ?",
            (cleaned, now, thread_id),
        )

    def touch(self, thread_id: str, title: str | None = None) -> None:
        now = utc_now_iso()
        if title:
            self._db.execute_write(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE thread_id = ?",
                (title, now, thread_id),
            )
        else:
            self._db.execute_write(
                "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )

    def list_conversations(self) -> list[ConversationInfo]:
        rows = self._db.query_all(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        )
        return [
            ConversationInfo(
                thread_id=row["thread_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def delete_conversation(self, thread_id: str) -> ConversationInfo | None:
        """Remove a chat session from the registry.

        Deletes ONLY the conversation entry. The shared document library,
        long-term preferences, tasks, and other threads are untouched. The
        LangGraph checkpoints for the thread are also left in place so a
        pending approval resumed through the Approval Center keeps working.
        """
        record = self.get_conversation(thread_id)
        if record is None:
            return None
        self._db.execute_write(
            "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        return record

    def get_conversation(self, thread_id: str) -> ConversationInfo | None:
        row = self._db.query_one(
            "SELECT * FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        if row is None:
            return None
        return ConversationInfo(
            thread_id=row["thread_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- pending operations -------------------------------------------------------
    def record_pending_operation(
        self, operation_id: str, thread_id: str, proposal: dict
    ) -> None:
        """Index a task proposal that is waiting for human approval."""
        self._db.execute_write(
            """
            INSERT OR IGNORE INTO pending_operations
                (operation_id, thread_id, proposal_json, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
            """,
            (operation_id, thread_id, json.dumps(proposal), utc_now_iso()),
        )

    def get_pending_operation(self, operation_id: str) -> PendingOperation | None:
        row = self._db.query_one(
            "SELECT * FROM pending_operations WHERE operation_id = ?",
            (operation_id,),
        )
        return self._pending_from_row(row) if row else None

    def list_pending_operations(
        self, status: str = "pending"
    ) -> list[PendingOperation]:
        rows = self._db.query_all(
            "SELECT * FROM pending_operations WHERE status = ? "
            "ORDER BY created_at DESC",
            (status,),
        )
        return [self._pending_from_row(row) for row in rows]

    def resolve_pending_operation(self, operation_id: str, status: str) -> None:
        """Move a pending operation to a resolved status.

        ``cancelled`` marks an approved run that was stopped before its
        task write (follow-up task, section 7) - reported honestly instead
        of "executed".
        """
        if status not in (
            "approved", "rejected", "executed", "failed", "cancelled",
            "pending",
        ):
            raise ValueError(f"Invalid pending-operation status: {status}")
        try:
            self._db.execute_write(
                "UPDATE pending_operations SET status = ?, resolved_at = ? "
                "WHERE operation_id = ?",
                (status, utc_now_iso(), operation_id),
            )
        except Exception:  # noqa: BLE001 - old-schema compatibility
            # Databases created before the 'cancelled' status carry a
            # narrower CHECK constraint. Recording the stop as 'failed'
            # preserves the honest "not executed" bookkeeping there.
            if status != "cancelled":
                raise
            self._db.execute_write(
                "UPDATE pending_operations SET status = ?, resolved_at = ? "
                "WHERE operation_id = ?",
                ("failed", utc_now_iso(), operation_id),
            )

    @staticmethod
    def _pending_from_row(row) -> PendingOperation:
        return PendingOperation(
            operation_id=row["operation_id"],
            thread_id=row["thread_id"],
            proposal=json.loads(row["proposal_json"]),
            status=row["status"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )
