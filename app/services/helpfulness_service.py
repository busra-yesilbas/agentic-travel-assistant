"""
Helpfulness / critique service.

Evaluates the quality of a generated travel plan against the original
user intent. Uses the LLM for nuanced critique, with a rule-based
fallback to ensure the pipeline never blocks.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.schemas.domain import (
    BudgetLevel,
    Critique,
    Itinerary,
    RankedHotel,
    TripIntent,
)
from app.services.llm import LLMProvider, parse_json_response
from app.services.prompt_manager import PromptManager

logger = get_logger(__name__)

# Budget label for matching in the itinerary text
_BUDGET_LABELS = {
    BudgetLevel.BUDGET: ["budget", "hostel", "affordable", "cheap"],
    BudgetLevel.MID: ["mid-range", "mid range", "moderate", "comfortable"],
    BudgetLevel.UPPER_MID: ["upper mid", "upscale", "nicer", "boutique"],
    BudgetLevel.LUXURY: ["luxury", "five star", "5-star", "exclusive", "premium"],
}


class HelpfulnessService:
    """
    Evaluates whether the generated plan meets the user's stated requirements.

    Combines rule-based heuristics with an optional LLM critique for
    nuanced quality signals.
    """

    def __init__(self, llm: LLMProvider, prompt_manager: PromptManager) -> None:
        self._llm = llm
        self._prompts = prompt_manager

    def evaluate(
        self,
        intent: TripIntent,
        ranked_hotels: list[RankedHotel],
        itinerary: Itinerary,
        final_answer: str,
    ) -> Critique:
        """
        Run the full critique pipeline.

        Tries the LLM critique first; falls back to rule-based evaluation
        if parsing fails.
        """
        try:
            return self._llm_critique(intent, ranked_hotels, itinerary, final_answer)
        except Exception as exc:
            logger.warning("helpfulness.llm_critique_failed", error=str(exc))
            return self._rule_based_critique(intent, ranked_hotels, itinerary, final_answer)

    def _llm_critique(
        self,
        intent: TripIntent,
        ranked_hotels: list[RankedHotel],
        itinerary: Itinerary,
        final_answer: str,
    ) -> Critique:
        itinerary_summary = itinerary.to_summary()[:600]
        hotel_recommendations = "; ".join(
            f"{rh.hotel.name} (score: {rh.score:.2f})" for rh in ranked_hotels[:3]
        )

        messages = self._prompts.get_messages(
            "critique",
            variables={
                "query": intent.to_summary(),
                "parsed_intent": f"City: {intent.city}, Days: {intent.days}, Budget: {intent.budget_level.value}, Style: {intent.travel_style.value}",
                "itinerary_summary": itinerary_summary,
                "hotel_recommendations": hotel_recommendations,
            },
        )

        response = self._llm.generate(messages)
        data = parse_json_response(response, context="critique")

        return Critique(
            budget_respected=bool(data.get("budget_respected", True)),
            duration_included=bool(data.get("duration_included", True)),
            activities_sufficient=bool(data.get("activities_sufficient", True)),
            hotel_alignment=bool(data.get("hotel_alignment", True)),
            assumptions_stated=bool(data.get("assumptions_stated", False)),
            overall_score=float(data.get("overall_score", 0.8)),
            flags=data.get("flags", []),
            suggestions=data.get("suggestions", []),
            approved=bool(data.get("approved", True)),
        )

    def _rule_based_critique(
        self,
        intent: TripIntent,
        ranked_hotels: list[RankedHotel],
        itinerary: Itinerary,
        final_answer: str,
    ) -> Critique:
        """Deterministic rule-based quality check."""
        flags = []
        suggestions = []
        score_components = []

        # 1. Budget respected — check that hotel price levels match budget
        budget_respected = True
        if ranked_hotels:
            budget_map = {
                BudgetLevel.BUDGET: 1,
                BudgetLevel.MID: 2,
                BudgetLevel.UPPER_MID: 3,
                BudgetLevel.LUXURY: 4,
            }
            target_level = budget_map.get(intent.budget_level, 2)
            avg_price = sum(rh.hotel.price_level for rh in ranked_hotels[:3]) / min(
                3, len(ranked_hotels)
            )
            if abs(avg_price - target_level) > 1.5:
                budget_respected = False
                flags.append("Hotel price levels do not closely match stated budget.")
                suggestions.append("Adjust hotel filters to better match the budget constraint.")
        score_components.append(1.0 if budget_respected else 0.3)

        # 2. Duration covered
        duration_included = len(itinerary.days) >= intent.days
        if not duration_included:
            flags.append(
                f"Itinerary covers only {len(itinerary.days)} of {intent.days} requested days."
            )
            suggestions.append("Extend the itinerary to cover all requested trip days.")
        score_components.append(1.0 if duration_included else 0.4)

        # 3. Activities sufficient — at least 2 activities mentioned per day
        activities_per_day = [
            sum([bool(d.morning), bool(d.afternoon), bool(d.evening)]) for d in itinerary.days
        ]
        activities_sufficient = (
            all(count >= 2 for count in activities_per_day) if activities_per_day else False
        )
        if not activities_sufficient:
            flags.append("Some days have fewer than 2 activities scheduled.")
            suggestions.append("Add more activities or restaurant recommendations for sparse days.")
        score_components.append(1.0 if activities_sufficient else 0.5)

        # 4. Hotel alignment — check ranked hotels match key interests
        hotel_alignment = True
        if ranked_hotels and intent.interests:
            top_hotel = ranked_hotels[0].hotel
            interest_set = {i.lower() for i in intent.interests}
            matched_features = {
                "museums" if top_hotel.near_museum else None,
                "nightlife" if top_hotel.near_nightlife else None,
                "transport" if top_hotel.near_public_transport else None,
                "romantic" if top_hotel.romantic else None,
                "family" if top_hotel.family_friendly else None,
            } - {None}
            if not interest_set.intersection(matched_features) and len(interest_set) > 0:
                hotel_alignment = False
                flags.append("Top hotel recommendation may not fully align with stated interests.")
                suggestions.append("Review hotel features against the stated interests.")
        score_components.append(1.0 if hotel_alignment else 0.6)

        # 5. Assumptions stated — check if final answer mentions any caveats
        caveats = ["assume", "assuming", "caveat", "note that", "please note", "book in advance"]
        assumptions_stated = any(c in final_answer.lower() for c in caveats)
        score_components.append(0.9 if assumptions_stated else 0.7)

        overall_score = sum(score_components) / len(score_components) if score_components else 0.75

        return Critique(
            budget_respected=budget_respected,
            duration_included=duration_included,
            activities_sufficient=activities_sufficient,
            hotel_alignment=hotel_alignment,
            assumptions_stated=assumptions_stated,
            overall_score=round(overall_score, 3),
            flags=flags,
            suggestions=suggestions,
            approved=overall_score >= 0.55,
        )
