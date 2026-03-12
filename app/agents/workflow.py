"""
Agent workflow orchestration.

Implements the TripGenie multi-stage planning pipeline as a simple
sequential state machine. Each stage is an independent agent that
reads from and writes to a shared TripPlanningState object.

Pipeline stages:
  1. IntentAgent      — Parse user query → TripIntent
  2. RetrievalAgent   — TripIntent → candidate hotels / attractions / restaurants
  3. RankingAgent     — candidates → ranked + scored hotels
  4. ItineraryAgent   — ranked hotels + attractions → day-by-day plan
  5. CritiqueAgent    — plan → quality assessment
  6. AnswerAgent      — all outputs → final natural language response
"""

from __future__ import annotations

import time
import uuid

import structlog

from app.agents.answer_agent import AnswerAgent
from app.agents.critique_agent import CritiqueAgent
from app.agents.intent_agent import IntentAgent
from app.agents.itinerary_agent import ItineraryAgent
from app.agents.ranking_agent import RankingAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.state import AgentProtocol, TripPlanningState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import get_registry
from app.schemas.requests import TripPlanningRequest
from app.services.dataset_service import get_dataset_service
from app.services.helpfulness_service import HelpfulnessService
from app.services.itinerary_service import ItineraryService
from app.services.llm import create_llm_provider
from app.services.prompt_manager import PromptManager
from app.services.ranking_service import RankingService
from app.services.retrieval_service import RetrievalService
from app.utils.text import sanitise_query

logger = get_logger(__name__)


class TripPlanningWorkflow:
    """
    Orchestrates the end-to-end trip planning agent pipeline.

    Instantiates all agents with their required services and runs them
    sequentially. Each agent is given the full state object and returns
    an enriched version.

    The workflow is deliberately fail-safe: if a non-critical stage
    fails, the pipeline continues with whatever state is available.
    """

    def __init__(self, force_mock: bool = False) -> None:
        settings = get_settings()
        dataset = get_dataset_service()

        # Shared services
        llm = create_llm_provider(force_mock=force_mock)
        prompt_manager = PromptManager()

        # Agent instantiation
        self._intent_agent = IntentAgent(llm=llm, prompt_manager=prompt_manager)
        self._retrieval_agent = RetrievalAgent(retrieval_service=RetrievalService(dataset=dataset))
        self._ranking_agent = RankingAgent(ranking_service=RankingService())
        self._itinerary_agent = ItineraryAgent(
            itinerary_service=ItineraryService(llm=llm, prompt_manager=prompt_manager)
        )
        self._critique_agent = CritiqueAgent(
            helpfulness_service=HelpfulnessService(llm=llm, prompt_manager=prompt_manager)
        )
        self._answer_agent = AnswerAgent(llm=llm, prompt_manager=prompt_manager)

        self._pipeline: list[tuple[str, AgentProtocol]] = [
            ("intent", self._intent_agent),
            ("retrieval", self._retrieval_agent),
            ("ranking", self._ranking_agent),
            ("itinerary", self._itinerary_agent),
            ("critique", self._critique_agent),
            ("answer", self._answer_agent),
        ]
        self._settings = settings

    async def run(self, request: TripPlanningRequest) -> TripPlanningState:
        """
        Execute the full planning pipeline for the given request.

        Args:
            request: Validated API request object.

        Returns:
            Completed TripPlanningState with all outputs populated.
        """
        request_id = str(uuid.uuid4())[:8]

        # Bind request context to all log lines in this call
        structlog.contextvars.bind_contextvars(request_id=request_id)

        state = TripPlanningState(
            request_id=request_id,
            user_query=sanitise_query(request.query),
            trip_request=request,
        )

        overall_start = time.perf_counter()
        registry = get_registry()
        registry.counter("workflow.requests_total").inc()

        logger.info(
            "workflow.start",
            request_id=request_id,
            query_length=len(request.query),
        )

        for stage_name, agent in self._pipeline:
            try:
                state = agent.run(state)
            except Exception as exc:
                # Unexpected exception that the agent didn't catch internally
                logger.error(
                    "workflow.stage_unhandled_error",
                    stage=stage_name,
                    error=str(exc),
                )
                state.record_error(stage_name, f"Unhandled: {exc}")
                # Continue — a partial result is better than nothing

        total_ms = (time.perf_counter() - overall_start) * 1000
        registry.histogram("workflow.latency_ms").observe(total_ms)

        if state.mock_llm_used:
            registry.counter("workflow.mock_llm_used").inc()

        logger.info(
            "workflow.complete",
            request_id=request_id,
            total_ms=round(total_ms, 1),
            stages_completed=len(state.stage_latencies),
            is_complete=state.is_complete,
            ranked_hotels=len(state.ranked_hotels),
        )

        # Add summary metadata
        state.metadata = {
            "llm_provider": "mock" if state.mock_llm_used else get_settings().llm_provider,
            "llm_calls": state.llm_calls,
            "mock_llm_used": state.mock_llm_used,
            "stage_errors": state.stage_errors,
            "candidate_count": len(state.candidate_hotels),
        }

        structlog.contextvars.clear_contextvars()
        return state
