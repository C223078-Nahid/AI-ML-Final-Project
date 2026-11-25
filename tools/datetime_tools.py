"""Current date/time as a model-invoked tool (follow-up task 4, section 7).

The deterministic utility (``core.datetime_util``) remains the single
reliable data source - the application clock in the configured
``APP_TIMEZONE``.  What changes is WHO decides when it is needed: the
model does, through function calling, instead of a hardcoded graph branch
routing every date question to a deterministic node.

The tool is read-only, takes no arguments, and returns the current date,
time, weekday, timezone, and ISO 8601 timestamp.  It never consults
documents or model training data.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from pydantic import BaseModel


class GetCurrentDatetimeInput(BaseModel):
    """The tool takes no arguments (kept explicit for schema clarity)."""

    pass


def make_datetime_tool():
    """Build the read-only get_current_datetime tool."""

    @tool("get_current_datetime", args_schema=GetCurrentDatetimeInput)
    def get_current_datetime() -> str:
        """Return the current date and time in the configured application timezone.

        Call this whenever the user asks about the current date, the current
        time, today's weekday, or anything that depends on "now".  Never
        answer such questions from your training data, and never look for
        the current date in the user's documents.  The response includes the
        date, the time, the weekday, the timezone and an ISO 8601 timestamp.
        """
        from core import datetime_util

        payload = datetime_util.datetime_payload()
        # Compact JSON: every field the model needs (ISO plus a
        # human-readable rendering), nothing else.
        return json.dumps(
            {
                "date": payload["date"],
                "date_text": payload["text"],
                "time": payload["time"],
                "weekday": payload["weekday"],
                "timezone": payload["timezone"],
                "iso_8601": payload["iso"],
                "source": payload["source"],
            },
            ensure_ascii=False,
        )

    return get_current_datetime
