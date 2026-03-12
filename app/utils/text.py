"""Text processing utilities."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "-", text).strip("-")


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max_length characters, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def sanitise_query(query: str) -> str:
    """Clean and normalise a user query string."""
    # Remove excessive whitespace
    query = re.sub(r"\s+", " ", query.strip())
    # Remove control characters
    query = "".join(ch for ch in query if unicodedata.category(ch) != "Cc")
    return query[:2000]  # Hard cap


def extract_numbers(text: str) -> list[int]:
    """Extract all integers from a text string."""
    return [int(m) for m in re.findall(r"\d+", text)]


def format_currency(amount: float, currency: str = "EUR") -> str:
    """Format a float as a currency string."""
    symbol = {"EUR": "€", "USD": "$", "GBP": "£"}.get(currency, currency)
    return f"{symbol}{amount:.2f}"


def bullet_list(items: list[str], indent: int = 0) -> str:
    """Format a list of strings as a markdown bullet list."""
    prefix = " " * indent
    return "\n".join(f"{prefix}- {item}" for item in items)
