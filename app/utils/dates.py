"""Date and time utility helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def today_str() -> str:
    """Return today's date as ISO format string."""
    return date.today().isoformat()


def add_days(start: str | date, days: int) -> str:
    """Return a date string that is `days` after the start date."""
    if isinstance(start, str):
        start = date.fromisoformat(start)
    return (start + timedelta(days=days)).isoformat()


def date_range(start: str | date, days: int) -> list[str]:
    """Return a list of date strings for a trip of `days` duration."""
    if isinstance(start, str):
        start = date.fromisoformat(start)
    return [(start + timedelta(days=i)).isoformat() for i in range(days)]


def next_month_start() -> str:
    """Return the first day of next month as an ISO date string."""
    today = date.today()
    if today.month == 12:
        first_next = date(today.year + 1, 1, 1)
    else:
        first_next = date(today.year, today.month + 1, 1)
    return first_next.isoformat()


def parse_date(text: str) -> date | None:
    """Try to parse a date string in common formats. Returns None on failure."""
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%B %d %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def duration_label(days: int) -> str:
    """Return a human-friendly duration label."""
    if days == 1:
        return "a one-day trip"
    elif days <= 3:
        return f"a {days}-day short break"
    elif days <= 7:
        return f"a {days}-day week-long trip"
    else:
        return f"a {days}-day extended stay"


def now_utc_iso() -> str:
    """Return the current UTC datetime as an ISO 8601 string."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
