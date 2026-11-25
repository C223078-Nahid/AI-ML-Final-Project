"""SQLite connection management for application data.

A single :class:`Database` instance is intended to live for the lifetime of
the process (Streamlit caches it with ``st.cache_resource``).  All writes go
through a re-entrant lock because Streamlit reruns and background threads
share the connection.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class Database:
    """Thin wrapper around a sqlite3 connection with serialized writes."""

    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")

    # -- properties ----------------------------------------------------------
    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    # -- helpers ---------------------------------------------------------------
    def execute_write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run a single write statement and commit atomically."""
        with self.lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def query_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self.lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self.lock:
            self._conn.close()
