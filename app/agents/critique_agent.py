"""
Critique agent.

Evaluates the quality and completeness of the generated travel plan
before the final answer is synthesised.
"""

from __future__ import annotations

import time

from app.agents.state import TripPlanningState
from app.core.logging import get_logger
from app.schemas.domain import Critique
from app.services.helpfulness_service import HelpfulnessService

logger = get_logger(__name__)


class CritiqueAgent:
    """
    Evaluates the quality of the travel plan against user requirements.

    Runs even if some earlier stages partially failed, using whatever
    state is available.
    """

    def __init__(self, helpfulness_service: HelpfulnessService) -> None:
        self._service = helpfulness_service

    def run(self, state: TripPlanningState) -> TripPlanningState:
        start = time.perf_counter()
        logger.info("agent.critique.start")

        if state.parsed_intent is None or state.itinerary is None:
            logger.warning("agent.critique.skipped_insufficient_state")
            state.critique = Critique(overall_score=0.5, approved=True)
            state.record_stage("critique", 0.0)
            return state

        try:
            state.critique = self._service.evaluate(
                intent=state.parsed_intent,
                ranked_hotels=state.ranked_hotels,
                itinerary=state.itinerary,
                final_answer=state.final_answer or "",
            )
            state.llm_calls += 1
            logger.info(
                "agent.critique.done",
                score=state.critique.overall_score,
                approved=state.critique.approved,
                flags=len(state.critique.flags),
            )
        except Exception as exc:
            logger.error("agent.critique.failed", error=str(exc))
            state.record_error("critique", str(exc))
            state.critique = Critique(
                overall_score=0.7,
                approved=True,
                flags=["Critique evaluation encountered an error."],
            )

        state.record_stage("critique", (time.perf_counter() - start) * 1000)
        return state
