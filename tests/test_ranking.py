"""Tests for the hotel ranking ML layer."""

from __future__ import annotations

import pytest

from app.ml.features import extract_features
from app.ml.ranker import HotelRanker
from app.schemas.domain import BudgetLevel, Hotel, TravelStyle, TripIntent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_hotels() -> list[Hotel]:
    return [
        Hotel(
            hotel_id="TEST001",
            city="Amsterdam",
            name="Budget Backpacker",
            price_level=1,
            avg_review_score=7.5,
            location_score=7.0,
            near_public_transport=True,
            budget=True,
        ),
        Hotel(
            hotel_id="TEST002",
            city="Amsterdam",
            name="Museum Quarter Inn",
            price_level=2,
            avg_review_score=8.6,
            location_score=8.8,
            near_museum=True,
            near_public_transport=True,
        ),
        Hotel(
            hotel_id="TEST003",
            city="Amsterdam",
            name="Canal Luxury Palace",
            price_level=4,
            avg_review_score=9.5,
            location_score=9.3,
            romantic=True,
            luxury=True,
            near_museum=True,
            near_public_transport=True,
        ),
        Hotel(
            hotel_id="TEST004",
            city="Amsterdam",
            name="Family Suites",
            price_level=2,
            avg_review_score=8.4,
            location_score=8.0,
            family_friendly=True,
            near_public_transport=True,
        ),
        Hotel(
            hotel_id="TEST005",
            city="Amsterdam",
            name="Nightlife Inn",
            price_level=2,
            avg_review_score=8.2,
            location_score=8.5,
            near_nightlife=True,
            near_public_transport=True,
        ),
    ]


@pytest.fixture
def mid_range_cultural_intent() -> TripIntent:
    return TripIntent(
        city="Amsterdam",
        days=4,
        travelers=2,
        budget_level=BudgetLevel.MID,
        interests=["museums", "canals", "culture"],
        travel_style=TravelStyle.CULTURAL,
    )


@pytest.fixture
def luxury_romantic_intent() -> TripIntent:
    return TripIntent(
        city="Amsterdam",
        days=4,
        travelers=2,
        budget_level=BudgetLevel.LUXURY,
        interests=["romantic", "luxury", "museums"],
        travel_style=TravelStyle.ROMANTIC,
    )


@pytest.fixture
def budget_solo_intent() -> TripIntent:
    return TripIntent(
        city="Amsterdam",
        days=3,
        travelers=1,
        budget_level=BudgetLevel.BUDGET,
        interests=["sightseeing"],
        travel_style=TravelStyle.CULTURAL,
    )


@pytest.fixture
def family_intent() -> TripIntent:
    return TripIntent(
        city="Amsterdam",
        days=5,
        travelers=4,
        budget_level=BudgetLevel.MID,
        interests=["family", "parks"],
        travel_style=TravelStyle.FAMILY,
    )


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------
class TestFeatureExtraction:
    def test_feature_vector_has_all_fields(self, sample_hotels, mid_range_cultural_intent) -> None:
        fv = extract_features(sample_hotels[1], mid_range_cultural_intent)
        assert hasattr(fv, "budget_match")
        assert hasattr(fv, "location_score")
        assert hasattr(fv, "review_score")
        assert hasattr(fv, "transport_match")
        assert hasattr(fv, "museum_affinity")
        assert hasattr(fv, "romantic_match")
        assert hasattr(fv, "family_match")

    def test_features_in_0_1_range(self, sample_hotels, mid_range_cultural_intent) -> None:
        for hotel in sample_hotels:
            fv = extract_features(hotel, mid_range_cultural_intent)
            for val in fv.to_array():
                assert 0.0 <= val <= 1.0, f"Feature out of range: {val}"

    def test_budget_match_perfect_for_mid_range(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        mid_hotel = sample_hotels[1]  # price_level=2
        fv = extract_features(mid_hotel, mid_range_cultural_intent)
        assert fv.budget_match == 1.0

    def test_budget_match_poor_for_luxury_vs_budget(
        self, sample_hotels, budget_solo_intent
    ) -> None:
        luxury_hotel = sample_hotels[2]  # price_level=4
        fv = extract_features(luxury_hotel, budget_solo_intent)
        assert fv.budget_match < 0.3

    def test_museum_affinity_high_when_near_museum(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        museum_hotel = sample_hotels[1]  # near_museum=True
        fv = extract_features(museum_hotel, mid_range_cultural_intent)
        assert fv.museum_affinity == 1.0

    def test_museum_affinity_low_when_not_near_museum(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        budget_hotel = sample_hotels[0]  # near_museum=False
        fv = extract_features(budget_hotel, mid_range_cultural_intent)
        assert fv.museum_affinity < 0.5

    def test_romantic_match_high_for_romantic_intent(
        self, sample_hotels, luxury_romantic_intent
    ) -> None:
        romantic_hotel = sample_hotels[2]  # romantic=True
        fv = extract_features(romantic_hotel, luxury_romantic_intent)
        assert fv.romantic_match == 1.0

    def test_family_match_high_for_family_intent(self, sample_hotels, family_intent) -> None:
        family_hotel = sample_hotels[3]  # family_friendly=True
        fv = extract_features(family_hotel, family_intent)
        assert fv.family_match == 1.0


# ---------------------------------------------------------------------------
# Ranker tests
# ---------------------------------------------------------------------------
class TestHotelRanker:
    def test_returns_correct_number_of_hotels(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=3)
        assert len(ranked) == 3

    def test_returns_all_hotels_when_top_k_larger(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=100)
        assert len(ranked) == len(sample_hotels)

    def test_hotels_sorted_by_score_descending(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=5)
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_numbers_are_sequential(self, sample_hotels, mid_range_cultural_intent) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=3)
        ranks = [r.rank for r in ranked]
        assert ranks == [1, 2, 3]

    def test_scores_in_valid_range(self, sample_hotels, mid_range_cultural_intent) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=5)
        for r in ranked:
            assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_museum_hotel_ranks_highly_for_cultural_intent(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=5)
        top_hotel_ids = {r.hotel.hotel_id for r in ranked[:2]}
        assert "TEST002" in top_hotel_ids or "TEST003" in top_hotel_ids

    def test_luxury_hotel_ranks_highly_for_luxury_intent(
        self, sample_hotels, luxury_romantic_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, luxury_romantic_intent, top_k=3)
        assert ranked[0].hotel.hotel_id == "TEST003"

    def test_budget_hotel_ranks_highly_for_budget_intent(
        self, sample_hotels, budget_solo_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, budget_solo_intent, top_k=3)
        assert ranked[0].hotel.hotel_id == "TEST001"

    def test_feature_contributions_sum_to_score(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=1)
        hotel = ranked[0]
        contrib_sum = sum(c.contribution for c in hotel.feature_contributions)
        assert abs(contrib_sum - hotel.score) < 0.001, (
            f"Contribution sum {contrib_sum:.4f} != score {hotel.score:.4f}"
        )

    def test_explanation_is_non_empty_string(
        self, sample_hotels, mid_range_cultural_intent
    ) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank(sample_hotels, mid_range_cultural_intent, top_k=3)
        for r in ranked:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 10

    def test_empty_candidates_returns_empty(self, mid_range_cultural_intent) -> None:
        ranker = HotelRanker()
        ranked = ranker.rank([], mid_range_cultural_intent, top_k=5)
        assert ranked == []

    def test_custom_weights_affect_ranking(self, sample_hotels, mid_range_cultural_intent) -> None:
        """Hotels should rank differently when weights are changed drastically."""
        ranker_default = HotelRanker()
        ranker_review_heavy = HotelRanker(feature_weights={"review_score": 1.0})

        ranker_default.rank(sample_hotels, mid_range_cultural_intent, top_k=5)
        ranked_review = ranker_review_heavy.rank(sample_hotels, mid_range_cultural_intent, top_k=5)

        # When only review score matters, the luxury hotel (9.5) should be first
        assert ranked_review[0].hotel.hotel_id == "TEST003"
