"""
API response models.

All API responses are strongly typed Pydantic models with clear
documentation of every field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.domain import Critique, Itinerary, RankedHotel, TripIntent


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str
    environment: str
    llm_provider: str
    timestamp: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "version": "0.1.0",
                    "environment": "development",
                    "llm_provider": "mock",
                    "timestamp": "2026-01-15T10:30:00Z",
                }
            ]
        }
    }


class TripPlanningResponse(BaseModel):
    """Response for POST /trip/plan."""

    request_id: str = Field(..., description="Unique identifier for this request")
    parsed_intent: TripIntent = Field(
        ..., description="Structured intent extracted from the user query"
    )
    ranked_hotels: list[RankedHotel] = Field(
        ..., description="Top hotel recommendations ranked by fit score"
    )
    itinerary: Itinerary = Field(..., description="Day-by-day trip itinerary")
    final_answer: str = Field(
        ..., description="Natural language response synthesising all recommendations"
    )
    critique: Critique = Field(..., description="Quality assessment of the generated plan")
    stage_latencies: dict[str, float] = Field(
        default_factory=dict,
        description="Elapsed time (ms) for each workflow stage",
    )
    total_latency_ms: float = Field(
        ..., description="Total end-to-end processing time in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata: LLM provider used, fallback flags, etc.",
    )


class EvalQueryResult(BaseModel):
    """Result for a single query in an evaluation run."""

    query_id: str
    query: str
    city: str
    latency_ms: float
    constraint_satisfaction: float
    recommendation_relevance: float
    itinerary_completeness: float
    helpfulness_score: float
    overall_score: float
    error: str | None = None


class EvalRunResponse(BaseModel):
    """Response for POST /eval/run."""

    num_queries: int
    successful: int
    failed: int
    avg_latency_ms: float
    avg_constraint_satisfaction: float
    avg_recommendation_relevance: float
    avg_itinerary_completeness: float
    avg_helpfulness_score: float
    avg_overall_score: float
    results: list[EvalQueryResult] = Field(default_factory=list)
    summary: str = ""
