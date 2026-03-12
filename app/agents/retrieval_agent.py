"""
Retrieval agent.

Queries the dataset for candidate hotels, attractions, and restaurants
based on the structured trip intent produced by the IntentAgent.
"""

from __future__ import annotations

import time

from app.agents.state import TripPlanningState
from app.core.logging import get_logger
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RetrievalAgent:
    """Retrieves candidate travel data for a given trip intent."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self._retrieval = retrieval_service

    def run(self, state: TripPlanningState) -> TripPlanningState:
        start = time.perf_counter()
        logger.info(
            "agent.retrieval.start",
            city=state.parsed_intent.city if state.parsed_intent else "unknown",
        )

        if state.parsed_intent is None:
            logger.warning("agent.retrieval.skipped_no_intent")
            state.record_stage("retrieval", 0.0)
            return state

        try:
            intent = state.parsed_intent
            state.candidate_hotels = self._retrieval.retrieve_hotels(intent)
            state.candidate_attractions = self._retrieval.retrieve_attractions(intent)
            state.candidate_restaurants = self._retrieval.retrieve_restaurants(intent)

            logger.info(
                "agent.retrieval.done",
                hotels=len(state.candidate_hotels),
                attractions=len(state.candidate_attractions),
                restaurants=len(state.candidate_restaurants),
            )
        except Exception as exc:
            logger.error("agent.retrieval.failed", error=str(exc))
            state.record_error("retrieval", str(exc))

        state.record_stage("retrieval", (time.perf_counter() - start) * 1000)
        return state
