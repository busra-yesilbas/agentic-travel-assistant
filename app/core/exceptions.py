"""
Custom exception hierarchy for TripGenie.

Using typed exceptions lets us handle different failure modes distinctly
in the API layer and return appropriate HTTP status codes with useful messages.
"""

from __future__ import annotations


class TripGenieError(Exception):
    """Base exception for all TripGenie errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------
class ConfigurationError(TripGenieError):
    """Raised when the application is misconfigured."""


# ---------------------------------------------------------------------------
# Data layer errors
# ---------------------------------------------------------------------------
class DataLoadError(TripGenieError):
    """Raised when a dataset cannot be loaded."""


class CityNotSupportedError(TripGenieError):
    """Raised when a requested city is not in the dataset."""

    def __init__(self, city: str, supported: list[str]) -> None:
        super().__init__(
            f"City '{city}' is not supported.",
            detail=f"Supported cities: {', '.join(supported)}",
        )
        self.city = city
        self.supported = supported


# ---------------------------------------------------------------------------
# LLM / provider errors
# ---------------------------------------------------------------------------
class LLMError(TripGenieError):
    """Raised when an LLM call fails."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed into the expected format."""

    def __init__(self, raw_response: str, expected_format: str) -> None:
        super().__init__(
            "Failed to parse LLM response.",
            detail=f"Expected {expected_format}. Got: {raw_response[:200]}",
        )
        self.raw_response = raw_response


class LLMTimeoutError(LLMError):
    """Raised when an LLM call exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Agent / workflow errors
# ---------------------------------------------------------------------------
class AgentError(TripGenieError):
    """Raised when an individual agent step fails."""

    def __init__(self, agent_name: str, message: str) -> None:
        super().__init__(f"[{agent_name}] {message}")
        self.agent_name = agent_name


class WorkflowError(TripGenieError):
    """Raised when the overall agent workflow fails."""


# ---------------------------------------------------------------------------
# API / validation errors
# ---------------------------------------------------------------------------
class ValidationError(TripGenieError):
    """Raised when request validation fails beyond Pydantic's checks."""


class RequestTooLargeError(TripGenieError):
    """Raised when a request payload exceeds limits."""
