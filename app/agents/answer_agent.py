"""
Answer synthesis agent.

Produces the final, polished natural language response that the user
sees. Uses the LLM to synthesise all pipeline outputs into a warm,
helpful, and actionable travel recommendation.
"""

from __future__ import annotations

import time

from app.agents.state import TripPlanningState
from app.core.exceptions import LLMParseError
from app.core.logging import get_logger
from app.services.llm import LLMProvider
from app.services.prompt_manager import PromptManager

logger = get_logger(__name__)


class AnswerAgent:
    """Synthesises a final natural language response from all pipeline outputs."""

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        self._llm = llm
        self._prompts = prompt_manager

    def run(self, state: TripPlanningState) -> TripPlanningState:
        start = time.perf_counter()
        logger.info("agent.answer.start")

        if state.parsed_intent is None:
            state.final_answer = self._generic_fallback()
            state.record_stage("answer", 0.0)
            return state

        try:
            state.final_answer = self._generate_answer(state)
            state.llm_calls += 1
            logger.info(
                "agent.answer.done",
                answer_length=len(state.final_answer),
            )
        except Exception as exc:
            logger.error("agent.answer.failed", error=str(exc))
            state.record_error("answer", str(exc))
            state.final_answer = self._structured_fallback(state)

        state.record_stage("answer", (time.perf_counter() - start) * 1000)
        return state

    def _generate_answer(self, state: TripPlanningState) -> str:
        intent = state.parsed_intent
        hotel_summary = self._format_hotels(state)
        itinerary_overview = self._format_itinerary(state)
        critique_notes = self._format_critique(state)

        messages = self._prompts.get_messages(
            "final_answer",
            variables={
                "query": state.user_query,
                "city": intent.city,
                "days": intent.days,
                "travelers": intent.travelers,
                "budget_level": intent.budget_level.value,
                "ranked_hotels": hotel_summary,
                "itinerary_overview": itinerary_overview,
                "critique_notes": critique_notes,
            },
        )

        response = self._llm.generate(messages)
        if response.is_mock:
            state.mock_llm_used = True

        # The final answer is plain prose, not JSON
        return response.content.strip()

    @staticmethod
    def _format_hotels(state: TripPlanningState) -> str:
        if not state.ranked_hotels:
            return "No hotels ranked."
        lines = []
        for rh in state.ranked_hotels[:3]:
            lines.append(
                f"  [{rh.rank}] {rh.hotel.name} ({rh.hotel.price_label}, "
                f"score {rh.score:.2f}) — {rh.explanation}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_itinerary(state: TripPlanningState) -> str:
        if state.itinerary is None:
            return "Itinerary not generated."
        lines = [f"Trip: {state.itinerary.trip_name}"]
        if state.itinerary.overview:
            lines.append(state.itinerary.overview)
        for day in state.itinerary.days[:3]:
            lines.append(f"Day {day.day} ({day.theme}): {day.morning[:60]}...")
        return "\n".join(lines)

    @staticmethod
    def _format_critique(state: TripPlanningState) -> str:
        if state.critique is None:
            return "No quality review available."
        parts = [f"Quality score: {state.critique.overall_score:.0%}"]
        if state.critique.flags:
            parts.append("Flags: " + "; ".join(state.critique.flags))
        if state.critique.suggestions:
            parts.append("Suggestions: " + "; ".join(state.critique.suggestions[:2]))
        return " | ".join(parts)

    @staticmethod
    def _structured_fallback(state: TripPlanningState) -> str:
        intent = state.parsed_intent
        if intent is None:
            return AnswerAgent._generic_fallback()

        hotel_names = (
            ", ".join(rh.hotel.name for rh in state.ranked_hotels[:3])
            if state.ranked_hotels
            else "hotels in your area"
        )
        return (
            f"Here's your {intent.days}-day {intent.city} plan for {intent.travelers} traveller(s). "
            f"Based on your {intent.budget_level.value} budget and interests in "
            f"{', '.join(intent.interests[:3]) or 'the city'}, "
            f"I've selected these top hotels: {hotel_names}. "
            f"The full day-by-day itinerary is included above. "
            f"Please book popular attractions in advance to avoid disappointment."
        )

    @staticmethod
    def _generic_fallback() -> str:
        return (
            "I'd be happy to help plan your trip! "
            "Please provide more details about your destination, dates, and preferences "
            "so I can create a personalised itinerary for you."
        )
