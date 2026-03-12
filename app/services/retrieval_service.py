"""
Retrieval service.

Filters candidates from the dataset based on the structured trip intent.
Returns a ranked candidate set for the scoring layer.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.domain import Attraction, BudgetLevel, Hotel, Restaurant, TripIntent
from app.services.dataset_service import DatasetService

logger = get_logger(__name__)


class RetrievalService:
    """
    Retrieves candidate hotels, attractions, and restaurants for a trip.

    Applies coarse filtering based on city and broad budget compatibility,
    then returns a candidate set for the ML ranking layer.
    """

    def __init__(self, dataset: DatasetService) -> None:
        self._dataset = dataset
        self._settings = get_settings()

    def retrieve_hotels(self, intent: TripIntent) -> list[Hotel]:
        """
        Retrieve candidate hotels for the given trip intent.

        Applies city filter first, then soft budget filter.
        Returns up to `max_candidates` hotels.
        """
        all_hotels = self._dataset.get_hotels(city=intent.city)

        if not all_hotels:
            logger.warning(
                "retrieval.no_hotels_for_city",
                city=intent.city,
                available=self._dataset.get_supported_cities(),
            )
            return []

        # Soft budget filter: exclude hotels that are wildly mismatched
        budget_filter = self._make_budget_filter(intent.budget_level)
        candidates = [h for h in all_hotels if budget_filter(h)]

        # If budget filter eliminates everything, relax it
        if not candidates:
            logger.debug("retrieval.budget_filter_relaxed", city=intent.city)
            candidates = all_hotels

        max_c = self._settings.retrieval_max_candidates
        result = candidates[:max_c]

        logger.info(
            "retrieval.hotels",
            city=intent.city,
            total_available=len(all_hotels),
            candidates=len(result),
            budget=intent.budget_level,
        )
        return result

    def retrieve_attractions(self, intent: TripIntent) -> list[Attraction]:
        """Retrieve relevant attractions for the trip intent."""
        all_attractions = self._dataset.get_attractions(city=intent.city)
        if not all_attractions:
            return []

        # Priority-sort by tag overlap with user interests
        interest_set = {i.lower() for i in intent.interests}
        scored = []
        for attraction in all_attractions:
            tag_set = {t.lower() for t in attraction.tags}
            overlap = len(interest_set & tag_set)
            scored.append((overlap, attraction.rating, attraction))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result = [item[2] for item in scored[:15]]

        logger.info(
            "retrieval.attractions",
            city=intent.city,
            total=len(all_attractions),
            returned=len(result),
        )
        return result

    def retrieve_restaurants(self, intent: TripIntent) -> list[Restaurant]:
        """Retrieve suitable restaurants for the trip intent."""
        all_restaurants = self._dataset.get_restaurants(city=intent.city)
        if not all_restaurants:
            return []

        filtered = []
        for restaurant in all_restaurants:
            # Family trip: prefer family-friendly options
            if (
                intent.travel_style.value == "family"
                and not restaurant.family_friendly
                and restaurant.rating < 9.0
            ):
                continue
            # Romantic: prefer romantic venues
            if intent.travel_style.value == "romantic":
                filtered.append((int(restaurant.romantic), restaurant.rating, restaurant))
                continue
            filtered.append((0, restaurant.rating, restaurant))

        filtered.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result = [item[2] for item in filtered[:8]]

        logger.info(
            "retrieval.restaurants",
            city=intent.city,
            total=len(all_restaurants),
            returned=len(result),
        )
        return result

    @staticmethod
    def _make_budget_filter(budget_level: BudgetLevel) -> Callable[[Hotel], bool]:
        """Return a predicate that filters hotels by broad budget compatibility."""
        level_map = {
            BudgetLevel.BUDGET: {1, 2},
            BudgetLevel.MID: {1, 2, 3},
            BudgetLevel.UPPER_MID: {2, 3, 4},
            BudgetLevel.LUXURY: {3, 4},
        }
        allowed_levels = level_map.get(budget_level, {1, 2, 3, 4})
        return lambda hotel: hotel.price_level in allowed_levels
