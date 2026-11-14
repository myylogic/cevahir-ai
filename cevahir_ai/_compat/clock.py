"""UTC timestamps in one form: RFC 3339 with microseconds and ``Z`` (they sort as text)."""
import re
from datetime import datetime, timedelta, timezone

_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_DURATION = re.compile(r"^(\d+)\s*(s|m|h|d|w)$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def utc_now():
    return datetime.now(timezone.utc).strftime(_FORMAT)


def parse_since(text, now=None):
    """``14d``, ``2h``, ``30m`` (that long ago) or a date/time (``2026-11-13``, RFC 3339) as a timestamp."""
    text = str(text).strip()
    match = _DURATION.match(text.lower())
    if match:
        current = datetime.strptime(now, _FORMAT).replace(tzinfo=timezone.utc) if now else datetime.now(timezone.utc)
        moment = current - timedelta(**{_UNITS[match.group(2)]: int(match.group(1))})
    else:
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"not a duration (14d, 2h, 30m) or a date: {text!r}") from None
        moment = moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    return moment.strftime(_FORMAT)
