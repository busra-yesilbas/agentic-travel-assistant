"""
API request models.

These are the Pydantic models that validate and document incoming
API payloads. They are separate from domain models to allow the API
contract to evolve independently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TripPlanningRequest(BaseModel):
    """
    Request body for POST /trip/plan.

    The `query` field is required; all other fields are optional overrides
    that take precedence over what the LLM extracts from the query text.
    """

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language travel request",
        examples=["I want a 4-day Amsterdam trip for a couple, mid-range budget, close to museums."],
    )
    city: str | None = Field(
        default=None,
        description="Override city extracted from query",
    )
    days: int | None = Field(
        default=None,
        ge=1,
        le=30,
        description="Override trip duration in days",
    )
    travelers: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override number of travelers",
    )
    budget_level: str | None = Field(
        default=None,
        description="Override budget level: budget | mid | upper_mid | luxury",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Explicit interests to merge with extracted ones",
        examples=[["museums", "food", "cycling"]],
    )
    style: str | None = Field(
        default=None,
        description="Override travel style: cultural | adventure | relaxation | nightlife | family | business | romantic",
    )
    use_llm: bool = Field(
        default=True,
        description="Whether to use the configured LLM (or fall back to mock)",
    )

    @field_validator("budget_level")
    @classmethod
    def validate_budget(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"budget", "mid", "upper_mid", "luxury"}
        if v.lower() not in allowed:
            raise ValueError(f"budget_level must be one of {allowed}")
        return v.lower()

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"cultural", "adventure", "relaxation", "nightlife", "family", "business", "romantic"}
        if v.lower() not in allowed:
            raise ValueError(f"style must be one of {allowed}")
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "I want a 4-day Amsterdam trip for a couple, mid-range budget, close to museums and public transport, with a canal cruise.",
                    "travelers": 2,
                    "interests": ["museums", "canal cruise"],
                    "use_llm": True,
                }
            ]
        }
    }


class EvalRunRequest(BaseModel):
    """Request body for POST /eval/run."""

    num_samples: int = Field(
        default=10,
        ge=1,
        le=10,
        description="Number of sample queries to evaluate (max 10)",
    )
    save_results: bool = Field(
        default=False,
        description="Whether to persist evaluation results to disk",
    )
    verbose: bool = Field(
        default=False,
        description="Include per-query result details in the response",
    )
