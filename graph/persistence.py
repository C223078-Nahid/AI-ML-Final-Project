"""LangGraph persistence: SQLite checkpointer (concepts C4, E2).

The checkpointer stores the full workflow state (messages, pending
proposals, routing position) for every thread, which is what makes
conversation memory, durable pause/resume, and restart recovery work.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_checkpointer(db_path: str) -> SqliteSaver:
    """Create (and initialize) a persistent SQLite checkpointer."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver
