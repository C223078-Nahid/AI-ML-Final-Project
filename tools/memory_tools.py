"""Long-term memory access (elective E3).

``get_user_preferences`` is a read-only tool.  Writing preferences is only
possible through the Settings page (explicit user action) - never through
the assistant or a model tool.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from core.models import UserPreference
from database.memory import MemoryRepository


def get_preferences(memory_repo: MemoryRepository) -> list[UserPreference]:
    """Deterministically load saved preferences (used to build prompts)."""
    return memory_repo.list_preferences()


def make_get_preferences_tool(memory_repo: MemoryRepository) -> Any:
    @tool("get_user_preferences")
    def get_user_preferences() -> str:
        """Read the user's explicitly saved long-term preferences.

        Use this when you need the user's saved defaults, for example their
        preferred task format or default task priority.
        """
        preferences = memory_repo.list_preferences()
        if not preferences:
            return "The user has not saved any preferences yet."
        lines = [f"- {pref.key}: {pref.value}" for pref in preferences]
        return "Saved user preferences:\n" + "\n".join(lines)

    return get_user_preferences
