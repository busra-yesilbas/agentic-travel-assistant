"""Tests for the evaluation framework."""

from __future__ import annotations

import pytest

from app.ml.eval_metrics import EvaluationMetrics
from app.schemas.domain import (
    BudgetLevel,
    Critique,
    Itinerary,
    ItineraryDay,
    RankedHotel,
    Hotel,
    TravelStyle,
    TripIntent,
    FeatureContribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def metrics() -> EvaluationMetrics:
    return EvaluationMetrics()


@pytest.fixture
def perfect_intent() -> TripIntent:
    return TripIntent(
        city="Amsterdam",
        days=4,
        travelers=2,
        budget_level=BudgetLevel.MID,
        interests=["museums", "canals"],
        travel_style=TravelStyle.CULTURAL,
    )


@pytest.fixture
def sample_hotel() -> Hotel:
    return Hotel(
        hotel_id="TEST001",
        city="Amsterdam",
        name="Museum Quarter Inn",
        price_level=2,
        avg_review_score=8.6,
        location_score=8.8,
        near_museum=True,
        near_public_transport=True,
    )


@pytest.fixture
def sample_ranked_hotel(sample_hotel) -> RankedHotel:
    return RankedHotel(
        hotel=sample_hotel,
        rank=1,
        score=0.82,
        feature_contributions=[
            FeatureContribution(feature="budget_match", raw_value=1.0, weight=0.25, contribution=0.25),
            FeatureContribution(feature="museum_affinity", raw_value=1.0, weight=0.08, contribution=0.08),
        ],
        explanation="Strong budget match and museum proximity.",
    )


@pytest.fixture
def complete_itinerary() -> Itinerary:
    return Itinerary(
        trip_name="Amsterdam Adventure",
        city="Amsterdam",
        total_days=4,
        overview="A wonderful 4-day Amsterdam trip.",
        days=[
            ItineraryDay(
                day=i + 1,
                theme=f"Day {i+1} Theme",
                morning=f"Morning activity day {i+1}",
                afternoon=f"Afternoon activity day {i+1}",
                evening=f"Evening activity day {i+1}",
            )
            for i in range(4)
        ],
        practical_tips=["Book museums ahead.", "Use the OV-chipkaart."],
        total_estimated_cost="€400 per person",
    )


# ---------------------------------------------------------------------------
# Constraint satisfaction tests
# ---------------------------------------------------------------------------
class TestConstraintSatisfaction:
    def test_perfect_match_returns_1(self, metrics, perfect_intent) -> None:
        score = metrics.constraint_satisfaction(
            intent=perfect_intent,
            expected_city="Amsterdam",
            expected_days=4,
            expected_budget="mid",
        )
        assert score == 1.0

    def test_wrong_city_penalises_score(self, metrics, perfect_intent) -> None:
        score = metrics.constraint_satisfaction(
            intent=perfect_intent,
            expected_city="Paris",
            expected_days=4,
            expected_budget="mid",
        )
        assert score < 0.7

    def test_wrong_budget_penalises_score(self, metrics, perfect_intent) -> None:
        score = metrics.constraint_satisfaction(
            intent=perfect_intent,
            expected_city="Amsterdam",
            expected_days=4,
            expected_budget="luxury",
        )
        assert score < 0.8

    def test_close_days_score_high(self, metrics, perfect_intent) -> None:
        # 4 days expected, 3 extracted → should still score well
        intent_3_days = perfect_intent.model_copy(update={"days": 3})
        score = metrics.constraint_satisfaction(
            intent=intent_3_days,
            expected_city="Amsterdam",
            expected_days=4,
            expected_budget="mid",
        )
        assert score >= 0.7

    def test_score_in_valid_range(self, metrics, perfect_intent) -> None:
        score = metrics.constraint_satisfaction(
            intent=perfect_intent,
            expected_city="Amsterdam",
            expected_days=4,
            expected_budget="mid",
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Recommendation relevance tests
# ---------------------------------------------------------------------------
class TestRecommendationRelevance:
    def test_relevant_hotel_scores_high(
        self, metrics, perfect_intent, sample_ranked_hotel
    ) -> None:
        score = metrics.recommendation_relevance(
            ranked_hotels=[sample_ranked_hotel],
            intent=perfect_intent,
            key_interests=["museums"],
        )
        assert score >= 0.5

    def test_empty_hotels_returns_zero(self, metrics, perfect_intent) -> None:
        score = metrics.recommendation_relevance(
            ranked_hotels=[],
            intent=perfect_intent,
            key_interests=["museums"],
        )
        assert score == 0.0

    def test_score_in_valid_range(
        self, metrics, perfect_intent, sample_ranked_hotel
    ) -> None:
        score = metrics.recommendation_relevance(
            ranked_hotels=[sample_ranked_hotel],
            intent=perfect_intent,
            key_interests=["museums"],
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Itinerary completeness tests
# ---------------------------------------------------------------------------
class TestItineraryCompleteness:
    def test_complete_itinerary_scores_high(
        self, metrics, complete_itinerary
    ) -> None:
        score = metrics.itinerary_completeness(
            itinerary=complete_itinerary,
            expected_days=4,
        )
        assert score >= 0.8

    def test_none_itinerary_returns_zero(self, metrics) -> None:
        score = metrics.itinerary_completeness(itinerary=None, expected_days=4)
        assert score == 0.0

    def test_partial_itinerary_scores_lower(self, metrics) -> None:
        partial = Itinerary(
            trip_name="Test",
            city="Amsterdam",
            total_days=4,
            days=[
                ItineraryDay(day=1, morning="Morning", afternoon="Afternoon")
            ],
        )
        score = metrics.itinerary_completeness(itinerary=partial, expected_days=4)
        assert score < 0.7

    def test_score_in_valid_range(self, metrics, complete_itinerary) -> None:
        score = metrics.itinerary_completeness(
            itinerary=complete_itinerary, expected_days=4
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Answer helpfulness tests
# ---------------------------------------------------------------------------
class TestAnswerHelpfulness:
    def test_long_specific_answer_scores_high(self, metrics, perfect_intent) -> None:
        answer = (
            "Amsterdam is a wonderful destination and I've put together a 4-day itinerary "
            "for you. I recommend the Museum Quarter Inn for its proximity to the Rijksmuseum "
            "and Van Gogh Museum. The hotel is well-rated and near public transport. "
            "For the itinerary: Day 1 starts with the Rijksmuseum, "
            "followed by a canal cruise in the afternoon. Day 2 focuses on the Van Gogh Museum "
            "and the Jordaan neighbourhood. I recommend booking museum tickets well in advance "
            "as they sell out weeks ahead. Day 3 is a day trip to Keukenhof tulip gardens. "
            "Day 4 gives you time to explore De Pijp market and enjoy a farewell dinner."
        )
        score = metrics.answer_helpfulness(final_answer=answer, intent=perfect_intent)
        assert score >= 0.7

    def test_empty_answer_returns_zero(self, metrics, perfect_intent) -> None:
        score = metrics.answer_helpfulness(final_answer="", intent=perfect_intent)
        assert score == 0.0

    def test_very_short_answer_scores_low(self, metrics, perfect_intent) -> None:
        score = metrics.answer_helpfulness(
            final_answer="Here is your trip.", intent=perfect_intent
        )
        assert score < 0.5

    def test_score_in_valid_range(self, metrics, perfect_intent) -> None:
        score = metrics.answer_helpfulness(
            final_answer="Amsterdam trip details here.", intent=perfect_intent
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Integration: full evaluation run
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_eval_run_completes() -> None:
    from app.services.experiment_service import ExperimentService

    service = ExperimentService()
    result = await service.run_evaluation(num_samples=2)

    assert result.num_queries == 2
    assert result.successful >= 1
    assert 0 <= result.avg_overall_score <= 1
    assert result.avg_latency_ms > 0
    assert result.summary != ""
