"""
Evaluation metrics for the TripGenie assistant.

Provides quantitative measures of plan quality across four dimensions:
  1. Constraint satisfaction — did we extract the right trip parameters?
  2. Recommendation relevance — do the hotels match stated preferences?
  3. Itinerary completeness — does the itinerary cover the full trip?
  4. Answer helpfulness — is the final answer useful and specific?

These metrics are designed to be runnable without ground-truth labels,
using proxy signals from the available structured data.
"""

from __future__ import annotations

from app.schemas.domain import BudgetLevel, Itinerary, RankedHotel, TripIntent


class EvaluationMetrics:
    """
    Computes evaluation metrics for a single trip planning result.

    All metrics return floats in [0, 1] where 1.0 is best.
    """

    def constraint_satisfaction(
        self,
        intent: TripIntent,
        expected_city: str,
        expected_days: int,
        expected_budget: str,
    ) -> float:
        """
        Measure how accurately the intent extraction captured the user's requirements.

        Checks city, days, and budget level against expected values.
        Each dimension contributes equally (1/3 each).
        """
        scores = []

        # City match
        city_match = intent.city.lower() == expected_city.lower() if expected_city else True
        scores.append(1.0 if city_match else 0.0)

        # Days match (allow ±1 day tolerance)
        if expected_days:
            day_diff = abs(intent.days - expected_days)
            day_score = 1.0 if day_diff == 0 else 0.7 if day_diff == 1 else 0.3
        else:
            day_score = 1.0
        scores.append(day_score)

        # Budget match
        if expected_budget:
            budget_map = {"budget": "budget", "mid": "mid", "upper_mid": "upper_mid", "luxury": "luxury"}
            expected_norm = budget_map.get(expected_budget.lower(), expected_budget)
            budget_match = intent.budget_level.value == expected_norm
            scores.append(1.0 if budget_match else 0.4)
        else:
            scores.append(1.0)

        return sum(scores) / len(scores)

    def recommendation_relevance(
        self,
        ranked_hotels: list[RankedHotel],
        intent: TripIntent,
        key_interests: list[str],
    ) -> float:
        """
        Measure how well the top hotel recommendations align with stated interests.

        Uses the top-3 hotels and checks feature alignment with interests.
        """
        if not ranked_hotels:
            return 0.0

        interest_set = {i.lower() for i in key_interests}
        top_hotels = ranked_hotels[:3]

        scores = []
        for rh in top_hotels:
            hotel = rh.hotel
            feature_matches = 0
            checks = 0

            if "museums" in interest_set or "art" in interest_set or "culture" in interest_set:
                feature_matches += int(hotel.near_museum)
                checks += 1
            if "nightlife" in interest_set or "party" in interest_set:
                feature_matches += int(hotel.near_nightlife)
                checks += 1
            if "transport" in interest_set or any("transport" in p.lower() for p in intent.accommodation_preferences):
                feature_matches += int(hotel.near_public_transport)
                checks += 1
            if "romantic" in interest_set or intent.travel_style.value == "romantic":
                feature_matches += int(hotel.romantic)
                checks += 1
            if "family" in interest_set or intent.travel_style.value == "family":
                feature_matches += int(hotel.family_friendly)
                checks += 1

            # Score alignment — higher ranked hotels should have higher scores
            score_bonus = min(rh.score, 1.0) * 0.3

            if checks > 0:
                hotel_score = (feature_matches / checks) * 0.7 + score_bonus
            else:
                hotel_score = 0.5 + score_bonus

            scores.append(hotel_score)

        # Weight top hotel more heavily
        if len(scores) == 1:
            return scores[0]
        elif len(scores) == 2:
            return scores[0] * 0.6 + scores[1] * 0.4
        else:
            return scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2

    def itinerary_completeness(
        self,
        itinerary: Itinerary | None,
        expected_days: int,
    ) -> float:
        """
        Measure how completely the itinerary covers the requested trip.

        Checks:
        - Day coverage ratio
        - Average activities per day
        - Presence of tips and practical info
        """
        if itinerary is None:
            return 0.0

        scores = []

        # Day coverage
        day_coverage = len(itinerary.days) / max(expected_days, 1)
        scores.append(min(day_coverage, 1.0))

        # Activities per day (should have at least morning + afternoon or evening)
        if itinerary.days:
            activities_per_day = [
                sum([bool(d.morning), bool(d.afternoon), bool(d.evening)])
                for d in itinerary.days
            ]
            avg_activities = sum(activities_per_day) / len(activities_per_day)
            activity_score = min(avg_activities / 3.0, 1.0)  # 3 activities = perfect
        else:
            activity_score = 0.0
        scores.append(activity_score)

        # Has practical tips
        has_tips = len(itinerary.practical_tips) >= 2
        scores.append(1.0 if has_tips else 0.5)

        # Has overview
        has_overview = len(itinerary.overview) > 20
        scores.append(1.0 if has_overview else 0.7)

        return sum(scores) / len(scores)

    def answer_helpfulness(
        self,
        final_answer: str,
        intent: TripIntent,
    ) -> float:
        """
        Heuristic measure of the final answer's helpfulness.

        Checks:
        - Minimum length (a very short answer is unhelpful)
        - Mentions the destination city
        - Mentions trip duration
        - Contains specific venue/activity names (length proxy)
        - Mentions booking or practical advice
        """
        if not final_answer:
            return 0.0

        scores = []
        text = final_answer.lower()

        # Length check: 100+ words is good, 300+ is excellent
        word_count = len(final_answer.split())
        length_score = min(word_count / 300, 1.0)
        scores.append(length_score)

        # Mentions the city
        city_mentioned = intent.city.lower() in text
        scores.append(1.0 if city_mentioned else 0.3)

        # Mentions trip duration
        days_mentioned = (
            str(intent.days) in final_answer
            or f"{intent.days}-day" in text
            or f"{intent.days} day" in text
        )
        scores.append(1.0 if days_mentioned else 0.5)

        # Contains practical advice signals
        practical_keywords = ["book", "reserve", "advance", "tip", "recommend", "avoid", "best time"]
        has_practical = any(kw in text for kw in practical_keywords)
        scores.append(1.0 if has_practical else 0.5)

        # Specificity: mentions at least a hotel name or attraction
        specificity_words = ["hotel", "museum", "restaurant", "tour", "canal", "eiffel", "colosseum"]
        is_specific = any(w in text for w in specificity_words)
        scores.append(1.0 if is_specific else 0.4)

        return sum(scores) / len(scores)
