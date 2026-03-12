"""
Ranking agent.

Scores and ranks candidate hotels retrieved by the RetrievalAgent.
Delegates to the ML ranking service and attaches scores and explanations
to the workflow state.
"""

from __future__ import annotations

import time

from app.agents.state import TripPlanningState
from app.core.logging import get_logger
from app.services.ranking_service import RankingService

logger = get_logger(__name__)


class RankingAgent:
    """Ranks candidate hotels by relevance to the trip intent."""

    def __init__(self, ranking_service: RankingService) -> None:
        self._ranking = ranking_service

    def run(self, state: TripPlanningState) -> TripPlanningState:
        start = time.perf_counter()
        logger.info(
            "agent.ranking.start",
            candidates=len(state.candidate_hotels),
        )

        if not state.candidate_hotels or state.parsed_intent is None:
            logger.warning("agent.ranking.skipped_no_candidates")
            state.record_stage("ranking", 0.0)
            return state

        try:
            state.ranked_hotels = self._ranking.rank_hotels(
                hotels=state.candidate_hotels,
                intent=state.parsed_intent,
            )
            logger.info(
                "agent.ranking.done",
                ranked=len(state.ranked_hotels),
                top_score=state.ranked_hotels[0].score if state.ranked_hotels else 0,
                top_hotel=state.ranked_hotels[0].hotel.name if state.ranked_hotels else "none",
            )
        except Exception as exc:
            logger.error("agent.ranking.failed", error=str(exc))
            state.record_error("ranking", str(exc))

        state.record_stage("ranking", (time.perf_counter() - start) * 1000)
        return state
