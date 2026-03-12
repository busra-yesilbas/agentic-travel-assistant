"""
Intent extraction agent.

Transforms the free-text user query into a structured TripIntent object.
Uses the LLM to parse the natural language request, with graceful fallback
to the MockLLMProvider's heuristic extraction if parsing fails.
"""

from __future__ import annotations

import contextlib
import time

from app.agents.state import TripPlanningState
from app.core.logging import get_logger
from app.schemas.domain import BudgetLevel, TravelStyle, TripIntent
from app.services.llm import LLMProvider, parse_json_response
from app.services.prompt_manager import PromptManager

logger = get_logger(__name__)


class IntentAgent:
    """
    Extracts structured trip intent from a natural language query.

    Merges LLM-extracted intent with any explicit overrides provided
    in the API request (city, days, travelers, budget_level, style).
    """

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        self._llm = llm
        self._prompts = prompt_manager

    def run(self, state: TripPlanningState) -> TripPlanningState:
        start = time.perf_counter()
        logger.info("agent.intent.start", request_id=state.request_id)

        try:
            intent = self._extract_intent(state)
            intent = self._apply_overrides(intent, state)
            state.parsed_intent = intent
            logger.info(
                "agent.intent.done",
                city=intent.city,
                days=intent.days,
                budget=intent.budget_level,
                travelers=intent.travelers,
            )
        except Exception as exc:
            logger.error("agent.intent.failed", error=str(exc))
            state.record_error("intent", str(exc))
            # Use sensible defaults so the pipeline can continue
            state.parsed_intent = self._default_intent(state)

        state.record_stage("intent", (time.perf_counter() - start) * 1000)
        return state

    def _extract_intent(self, state: TripPlanningState) -> TripIntent:
        """Call the LLM to parse the user query."""
        req = state.trip_request
        messages = self._prompts.get_messages(
            "intent_extraction",
            variables={
                "query": state.user_query,
                "city": req.city or "not specified",
                "days": req.days or "not specified",
                "travelers": req.travelers or "not specified",
                "budget_level": req.budget_level or "not specified",
                "interests": ", ".join(req.interests) if req.interests else "not specified",
                "style": req.style or "not specified",
            },
        )

        response = self._llm.generate(messages)
        state.llm_calls += 1
        if response.is_mock:
            state.mock_llm_used = True

        data = parse_json_response(response, context="intent_extraction")
        return self._dict_to_intent(data)

    def _dict_to_intent(self, data: dict) -> TripIntent:
        """Convert the parsed JSON dict to a TripIntent domain object."""
        try:
            budget = BudgetLevel(data.get("budget_level", "mid"))
        except ValueError:
            budget = BudgetLevel.MID

        try:
            style = TravelStyle(data.get("travel_style", "cultural"))
        except ValueError:
            style = TravelStyle.CULTURAL

        return TripIntent(
            city=data.get("city", "Amsterdam"),
            days=int(data.get("days", 3)),
            travelers=int(data.get("travelers", 2)),
            budget_level=budget,
            interests=list(data.get("interests", [])),
            travel_style=style,
            accommodation_preferences=list(data.get("accommodation_preferences", [])),
            special_requests=list(data.get("special_requests", [])),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
        )

    def _apply_overrides(self, intent: TripIntent, state: TripPlanningState) -> TripIntent:
        """Apply explicit request-level overrides to the LLM-extracted intent."""
        req = state.trip_request

        overrides: dict = {}
        if req.city:
            overrides["city"] = req.city
        if req.days:
            overrides["days"] = req.days
        if req.travelers:
            overrides["travelers"] = req.travelers
        if req.budget_level:
            with contextlib.suppress(ValueError):
                overrides["budget_level"] = BudgetLevel(req.budget_level)
        if req.style:
            with contextlib.suppress(ValueError):
                overrides["travel_style"] = TravelStyle(req.style)
        if req.interests:
            merged = list(set(intent.interests + req.interests))
            overrides["interests"] = merged

        if overrides:
            return intent.model_copy(update=overrides)
        return intent

    @staticmethod
    def _default_intent(state: TripPlanningState) -> TripIntent:
        """Produce a minimal intent when extraction fails completely."""
        req = state.trip_request
        return TripIntent(
            city=req.city or "Amsterdam",
            days=req.days or 3,
            travelers=req.travelers or 2,
            budget_level=BudgetLevel(req.budget_level or "mid"),
            interests=req.interests or [],
        )
