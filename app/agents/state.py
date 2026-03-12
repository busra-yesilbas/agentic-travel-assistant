"""
Shared workflow state.

TripPlanningState is the single mutable object that flows through the
entire agent pipeline. Each agent reads from it and writes its outputs
back, building up a richer result at each stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.schemas.domain import (
    Attraction,
    Critique,
    Hotel,
    Itinerary,
    RankedHotel,
    Restaurant,
    TripIntent,
)
from app.schemas.requests import TripPlanningRequest


@runtime_checkable
class AgentProtocol(Protocol):
    """Structural protocol that every pipeline agent must satisfy."""

    def run(self, state: TripPlanningState) -> TripPlanningState: ...


@dataclass
class TripPlanningState:
    """
    Mutable state object shared across all workflow agents.

    Fields are populated progressively as the pipeline advances.
    None values indicate that a stage has not yet run.
    """

    # --- Input ---
    request_id: str
    user_query: str
    trip_request: TripPlanningRequest

    # --- Stage 1: Intent extraction ---
    parsed_intent: TripIntent | None = None

    # --- Stage 2: Retrieval ---
    candidate_hotels: list[Hotel] = field(default_factory=list)
    candidate_attractions: list[Attraction] = field(default_factory=list)
    candidate_restaurants: list[Restaurant] = field(default_factory=list)

    # --- Stage 3: Ranking ---
    ranked_hotels: list[RankedHotel] = field(default_factory=list)

    # --- Stage 4: Itinerary ---
    itinerary: Itinerary | None = None

    # --- Stage 5: Critique ---
    critique: Critique | None = None

    # --- Stage 6: Final answer ---
    final_answer: str | None = None

    # --- Observability ---
    stage_latencies: dict[str, float] = field(default_factory=dict)
    stage_errors: dict[str, str] = field(default_factory=dict)
    llm_calls: int = 0
    mock_llm_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_stage(self, stage: str, latency_ms: float) -> None:
        """Record latency for a completed stage."""
        self.stage_latencies[stage] = round(latency_ms, 2)

    def record_error(self, stage: str, error: str) -> None:
        """Record a non-fatal error from a stage."""
        self.stage_errors[stage] = error

    @property
    def total_latency_ms(self) -> float:
        return sum(self.stage_latencies.values())

    @property
    def is_complete(self) -> bool:
        """True if all mandatory stages have produced output."""
        return all(
            [
                self.parsed_intent is not None,
                len(self.ranked_hotels) > 0,
                self.itinerary is not None,
                self.final_answer is not None,
            ]
        )
