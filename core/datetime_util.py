"""Deterministic current date/time utility (follow-up task, section 5.2 C).

"Why does this exist?" - the assistant previously answered "What day is
today?" from the model's training data (stale or wrong), and after the
retrieval-first fix the same question was hijacked into the document branch
and answered with a document abstention. Both are wrong: current date/time
is *dynamic application information*.

Rules (from the task):
- The value comes from the application clock in the configured timezone
  (``APP_TIMEZONE``, default ``Asia/Dhaka``) - never from model training
  data, never from uploaded documents, never hardcoded.
- The response is produced deterministically by this module; no model call
  is involved, so date/time answers cannot be blocked by the
  document-evidence abstention mechanism.
- Tests freeze the clock by monkeypatching ``now()``.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

# Human-friendly date/time rendering (en-GB style: 24 November 2026).
_DATE_FORMAT = "%A, %d %B %Y"
_TIME_FORMAT = "%H:%M"
_TIME_WITH_SECONDS = "%H:%M:%S"


def now(timezone_name: str | None = None) -> datetime:
    """Timezone-aware current datetime (UTC as the safe fallback)."""
    from config.settings import get_settings

    name = timezone_name or get_settings().app_timezone
    try:
        return datetime.now(ZoneInfo(name))
    except Exception:  # noqa: BLE001 - an invalid zone must never crash a chat
        return datetime.now(ZoneInfo("UTC"))


def format_datetime_response(
    dt: datetime | None = None,
    timezone_name: str | None = None,
    include_time: bool = True,
) -> str:
    """Render the deterministic answer text for a date/time question.

    ``include_time`` is decided by the caller from the question shape: a
    pure "what day is today?" still shows the time (harmless and useful),
    so it defaults to True and keeps the response format stable.
    """
    from config.settings import get_settings

    zone = timezone_name or get_settings().app_timezone
    current = dt if dt is not None else now(zone)
    date_part = current.strftime(_DATE_FORMAT)
    zone_label = current.tzname() or zone
    if include_time:
        time_part = current.strftime(_TIME_FORMAT)
        return (
            f"Today is {date_part}. The current time is {time_part} "
            f"({zone_label}, {zone})."
        )
    return f"Today is {date_part} ({zone})."


def datetime_payload(
    dt: datetime | None = None,
    timezone_name: str | None = None,
) -> dict:
    """Structured payload for the ``datetime_response`` graph node."""
    from config.settings import get_settings

    zone = timezone_name or get_settings().app_timezone
    current = dt if dt is not None else now(zone)
    return {
        "kind": "datetime",
        "text": format_datetime_response(current, zone),
        "iso": current.isoformat(timespec="seconds"),
        "date": current.date().isoformat(),
        "weekday": current.strftime("%A"),
        "time": current.strftime(_TIME_WITH_SECONDS),
        "timezone": zone,
        "source": "application clock",
    }
