"""
Itinerary service.

Generates day-by-day trip itineraries using the LLM layer.
Handles prompt construction, LLM call, response parsing, and
fallback to a deterministic template if parsing fails.
"""

from __future__ import annotations

from app.core.exceptions import LLMParseError
from app.core.logging import get_logger
from app.schemas.domain import (
    Attraction,
    Itinerary,
    ItineraryDay,
    RankedHotel,
    Restaurant,
    TripIntent,
)
from app.services.llm import LLMProvider, parse_json_response
from app.services.prompt_manager import PromptManager

logger = get_logger(__name__)


class ItineraryService:
    """Generates trip itineraries via the LLM provider."""

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        self._llm = llm
        self._prompts = prompt_manager

    def generate(
        self,
        intent: TripIntent,
        ranked_hotels: list[RankedHotel],
        attractions: list[Attraction],
        restaurants: list[Restaurant],
    ) -> Itinerary:
        """
        Generate a day-by-day itinerary.

        Calls the LLM with a structured prompt. Falls back to a deterministic
        template if the LLM response cannot be parsed.
        """
        hotel_summary = self._format_hotels(ranked_hotels)
        attraction_summary = self._format_attractions(attractions)
        restaurant_summary = self._format_restaurants(restaurants)

        messages = self._prompts.get_messages(
            "itinerary_generation",
            variables={
                "city": intent.city,
                "days": intent.days,
                "travelers": intent.travelers,
                "budget_level": intent.budget_level.value,
                "interests": ", ".join(intent.interests) or "general sightseeing",
                "travel_style": intent.travel_style.value,
                "special_requests": ", ".join(intent.special_requests) or "none",
                "ranked_hotels": hotel_summary,
                "attractions": attraction_summary,
                "restaurants": restaurant_summary,
            },
        )

        try:
            response = self._llm.generate(messages)
            data = parse_json_response(response, context="itinerary")
            itinerary = self._parse_itinerary(data, intent)
            logger.info(
                "itinerary.generated",
                city=intent.city,
                days=intent.days,
                is_mock=response.is_mock,
            )
            return itinerary
        except (LLMParseError, Exception) as exc:
            logger.warning("itinerary.parse_failed_using_fallback", error=str(exc))
            return self._fallback_itinerary(intent, ranked_hotels)

    def _parse_itinerary(self, data: dict, intent: TripIntent) -> Itinerary:
        """Convert LLM JSON response into an Itinerary domain object."""
        raw_days = data.get("days", [])
        days = []
        for i, day_data in enumerate(raw_days):
            if isinstance(day_data, dict):
                days.append(
                    ItineraryDay(
                        day=day_data.get("day", i + 1),
                        theme=day_data.get("theme", ""),
                        morning=day_data.get("morning", ""),
                        afternoon=day_data.get("afternoon", ""),
                        evening=day_data.get("evening", ""),
                        recommended_hotel=day_data.get("recommended_hotel", ""),
                        transport_tip=day_data.get("transport_tip", ""),
                        estimated_daily_budget=day_data.get("estimated_daily_budget", ""),
                    )
                )

        return Itinerary(
            trip_name=data.get("trip_name", f"{intent.city} Adventure"),
            city=intent.city,
            total_days=intent.days,
            overview=data.get("overview", ""),
            days=days,
            total_estimated_cost=data.get("total_estimated_cost", ""),
            practical_tips=data.get("practical_tips", []),
            best_time_to_visit=data.get("best_time_to_visit", ""),
        )

    def _fallback_itinerary(
        self, intent: TripIntent, ranked_hotels: list[RankedHotel]
    ) -> Itinerary:
        """Produce a minimal but valid itinerary when LLM parsing fails."""
        hotel_name = ranked_hotels[0].hotel.name if ranked_hotels else "your hotel"
        days = []
        for d in range(1, intent.days + 1):
            days.append(
                ItineraryDay(
                    day=d,
                    theme="Exploration",
                    morning=f"Day {d}: Morning — explore the {intent.city} city centre.",
                    afternoon="Afternoon — visit a local museum or attraction.",
                    evening="Evening — dine at a recommended local restaurant.",
                    recommended_hotel=hotel_name,
                    transport_tip="Use the public transport network to get around.",
                    estimated_daily_budget="€80-120 per person",
                )
            )
        return Itinerary(
            trip_name=f"{intent.city} Getaway",
            city=intent.city,
            total_days=intent.days,
            overview=f"A {intent.days}-day trip to {intent.city} for {intent.travelers} traveller(s).",
            days=days,
            total_estimated_cost="Varies",
            practical_tips=["Book attractions in advance.", "Use public transport."],
        )

    @staticmethod
    def _format_hotels(ranked_hotels: list[RankedHotel]) -> str:
        lines = []
        for rh in ranked_hotels[:5]:
            lines.append(
                f"  [{rh.rank}] {rh.hotel.name} (Score: {rh.score:.2f}) — {rh.explanation}"
            )
        return "\n".join(lines) if lines else "No hotels ranked yet."

    @staticmethod
    def _format_attractions(attractions: list[Attraction]) -> str:
        lines = []
        for a in attractions[:8]:
            lines.append(f"  - {a.name} ({a.category}, {a.duration_hours}h) — {a.description[:80]}")
        return "\n".join(lines) if lines else "No attractions data available."

    @staticmethod
    def _format_restaurants(restaurants: list[Restaurant]) -> str:
        lines = []
        for r in restaurants[:6]:
            lines.append(f"  - {r.name} ({r.cuisine}) — {r.description[:80]}")
        return "\n".join(lines) if lines else "No restaurant data available."
