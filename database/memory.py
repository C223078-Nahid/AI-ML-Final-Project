"""Long-term memory repository (elective E3).

Preferences are only stored when the user explicitly saves them from the
Settings page.  Secret-like keys are refused so credentials never end up in
long-term memory.
"""

from __future__ import annotations

import re

from database.connection import Database
from core.models import UserPreference, new_id, utc_now_iso

# Keys that look like credentials must never be stored in long-term memory.
_FORBIDDEN_KEY_FRAGMENTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "private_key",
)

# Values that look like raw credentials (OpenAI/Anthropic/GitHub/AWS style).
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"sk-[A-Za-z0-9_-]{16,}",
        r"sk-ant-[A-Za-z0-9_-]{16,}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"Bearer\s+[A-Za-z0-9._-]{20,}",
    )
)

MAX_KEY_LENGTH = 60
MAX_VALUE_LENGTH = 500


class PreferenceError(Exception):
    """Raised when a preference is rejected."""


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- validation -----------------------------------------------------------
    @staticmethod
    def _validate(key: str, value: str) -> tuple[str, str]:
        key_clean = " ".join(key.split())
        value_clean = value.strip()
        if not key_clean:
            raise PreferenceError("Preference key cannot be empty.")
        if len(key_clean) > MAX_KEY_LENGTH:
            raise PreferenceError(
                f"Preference key must be at most {MAX_KEY_LENGTH} characters."
            )
        if not value_clean:
            raise PreferenceError("Preference value cannot be empty.")
        if len(value_clean) > MAX_VALUE_LENGTH:
            raise PreferenceError(
                f"Preference value must be at most {MAX_VALUE_LENGTH} characters."
            )
        lowered_key = key_clean.lower()
        if any(fragment in lowered_key for fragment in _FORBIDDEN_KEY_FRAGMENTS):
            raise PreferenceError(
                "This looks like a credential. Secrets are never stored in "
                "long-term memory."
            )
        if any(pattern.search(value_clean) for pattern in _SECRET_VALUE_PATTERNS):
            raise PreferenceError(
                "This value looks like an API key or token. Secrets are never "
                "stored in long-term memory."
            )
        return key_clean, value_clean

    # -- CRUD ---------------------------------------------------------------------
    def save_preference(self, key: str, value: str) -> UserPreference:
        """Insert or update (upsert) a preference identified by ``key``."""
        key_clean, value_clean = self._validate(key, value)
        existing = self._db.query_one(
            "SELECT * FROM user_memory WHERE key = ?", (key_clean,)
        )
        now = utc_now_iso()
        if existing is None:
            record = UserPreference(
                id=new_id(), key=key_clean, value=value_clean,
                created_at=now, updated_at=now,
            )
            self._db.execute_write(
                "INSERT INTO user_memory (id, key, value, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (record.id, record.key, record.value, record.created_at, record.updated_at),
            )
        else:
            record = UserPreference(
                id=existing["id"], key=key_clean, value=value_clean,
                created_at=existing["created_at"], updated_at=now,
            )
            self._db.execute_write(
                "UPDATE user_memory SET value = ?, updated_at = ? WHERE key = ?",
                (value_clean, now, key_clean),
            )
        return record

    @staticmethod
    def _row_to_record(row) -> UserPreference:
        return UserPreference(
            id=row["id"], key=row["key"], value=row["value"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def list_preferences(self) -> list[UserPreference]:
        rows = self._db.query_all(
            "SELECT * FROM user_memory ORDER BY key COLLATE NOCASE"
        )
        return [self._row_to_record(row) for row in rows]

    def find(self, key: str) -> UserPreference | None:
        row = self._db.query_one(
            "SELECT * FROM user_memory WHERE key = ? COLLATE NOCASE", (key.strip(),)
        )
        return self._row_to_record(row) if row else None

    def delete_preference(self, memory_id: str) -> bool:
        cursor = self._db.execute_write(
            "DELETE FROM user_memory WHERE id = ?", (memory_id,)
        )
        return cursor.rowcount > 0
