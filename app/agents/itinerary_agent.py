"""
Itinerary agent.

Generates the day-by-day trip plan using the LLM, enriched with ranked
hotel and attraction data from earlier pipeline stages.
"""

from __future__ import annotations

import time

from app.agents.state import TripPlanningState
from app.core.logging import get_logger
from app.services.itinerary_service import ItineraryService

logger = get_logger(__name__)


class ItineraryAgent:
    """Generates a day-by-day travel itinerary."""

    def __init__(self, itinerary_service: ItineraryService) -> None:
        self._service = itinerary_service

    def run(self, state: TripPlanningState) -> TripPlanningState:
        start = time.perf_counter()
        logger.info(
            "agent.itinerary.start",
            city=state.parsed_intent.city if state.parsed_intent else "unknown",
            days=state.parsed_intent.days if state.parsed_intent else 0,
        )

        if state.parsed_intent is None:
            logger.warning("agent.itinerary.skipped_no_intent")
            state.record_stage("itinerary", 0.0)
            return state

        try:
            state.itinerary = self._service.generate(
                intent=state.parsed_intent,
                ranked_hotels=state.ranked_hotels,
                attractions=state.candidate_attractions,
                restaurants=state.candidate_restaurants,
            )
            state.llm_calls += 1
            logger.info(
                "agent.itinerary.done",
                trip_name=state.itinerary.trip_name,
                days=len(state.itinerary.days),
            )
        except Exception as exc:
            logger.error("agent.itinerary.failed", error=str(exc))
            state.record_error("itinerary", str(exc))

        state.record_stage("itinerary", (time.perf_counter() - start) * 1000)
        return state
