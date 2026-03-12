"""
Feature engineering for hotel ranking.

Each feature function takes a Hotel and a TripIntent and returns a float
in [0, 1] representing how well the hotel satisfies that dimension of the
user's requirements.

The feature set is deliberately interpretable: each feature has a clear
real-world meaning and contributes an explainable signal to the final score.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.domain import BudgetLevel, Hotel, TripIntent


@dataclass
class FeatureVector:
    """Computed feature values for a single (hotel, intent) pair."""

    hotel_id: str
    budget_match: float
    location_score: float
    review_score: float
    transport_match: float
    museum_affinity: float
    romantic_match: float
    family_match: float
    nightlife_match: float
    business_match: float

    def to_dict(self) -> dict[str, float]:
        return {
            "budget_match": self.budget_match,
            "location_score": self.location_score,
            "review_score": self.review_score,
            "transport_match": self.transport_match,
            "museum_affinity": self.museum_affinity,
            "romantic_match": self.romantic_match,
            "family_match": self.family_match,
            "nightlife_match": self.nightlife_match,
            "business_match": self.business_match,
        }

    def to_array(self) -> list[float]:
        return list(self.to_dict().values())


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def budget_match(hotel: Hotel, intent: TripIntent) -> float:
    """
    Score how closely the hotel's price level aligns with the user's budget.

    Perfect match → 1.0
    One level off → 0.65
    Two levels off → 0.25
    Three levels off → 0.0
    """
    target_map = {
        BudgetLevel.BUDGET: 1,
        BudgetLevel.MID: 2,
        BudgetLevel.UPPER_MID: 3,
        BudgetLevel.LUXURY: 4,
    }
    target = target_map.get(intent.budget_level, 2)
    diff = abs(hotel.price_level - target)
    scores = {0: 1.0, 1: 0.65, 2: 0.25, 3: 0.0}
    return scores.get(diff, 0.0)


def location_score_feature(hotel: Hotel, intent: TripIntent) -> float:
    """Normalise the hotel's location score to [0, 1]."""
    return hotel.location_score / 10.0


def review_score_feature(hotel: Hotel, intent: TripIntent) -> float:
    """Normalise the hotel's average review score to [0, 1]."""
    return hotel.avg_review_score / 10.0


def transport_match(hotel: Hotel, intent: TripIntent) -> float:
    """
    Score transport accessibility.

    Full score if the user cares about transport and the hotel is near it.
    Neutral (0.5) if transport isn't a stated interest.
    """
    transport_interests = {"transport", "metro", "tram", "bus", "public transport", "commute"}
    user_cares = bool(transport_interests & {i.lower() for i in intent.interests})
    user_cares = user_cares or any(
        p in ["near public transport", "easy transport"] for p in intent.accommodation_preferences
    )

    if hotel.near_public_transport:
        return 1.0
    if user_cares:
        return 0.1  # Strongly penalise if user needs transport and hotel lacks it
    return 0.5  # Neutral — transport not critical for this trip


def museum_affinity(hotel: Hotel, intent: TripIntent) -> float:
    """Score the hotel's proximity to museums against the user's cultural interest."""
    museum_interests = {"museums", "museum", "art", "gallery", "galleries", "culture", "history"}
    user_cares = bool(museum_interests & {i.lower() for i in intent.interests})

    if not user_cares:
        return 0.5  # Neutral

    return 1.0 if hotel.near_museum else 0.15


def romantic_match(hotel: Hotel, intent: TripIntent) -> float:
    """Score romantic suitability against the user's travel style."""
    from app.schemas.domain import TravelStyle

    romantic_signals = {
        TravelStyle.ROMANTIC,
    }
    romantic_interests = {"romantic", "romance", "honeymoon", "couples", "anniversary"}

    user_wants_romantic = (
        intent.travel_style in romantic_signals
        or bool(romantic_interests & {i.lower() for i in intent.interests})
        or any("romantic" in p.lower() for p in intent.accommodation_preferences)
    )

    if not user_wants_romantic:
        return 0.5  # Neutral

    return 1.0 if hotel.romantic else 0.2


def family_match(hotel: Hotel, intent: TripIntent) -> float:
    """Score family suitability for family trips."""
    from app.schemas.domain import TravelStyle

    family_signals = {TravelStyle.FAMILY}
    family_interests = {"family", "kids", "children", "toddler", "baby"}

    user_has_family = (
        intent.travel_style in family_signals
        or bool(family_interests & {i.lower() for i in intent.interests})
        or intent.travelers >= 3
    )

    if not user_has_family:
        return 0.5  # Neutral

    return 1.0 if hotel.family_friendly else 0.2


def nightlife_match(hotel: Hotel, intent: TripIntent) -> float:
    """Score nightlife proximity for travellers seeking the evening scene."""
    from app.schemas.domain import TravelStyle

    nightlife_interests = {"nightlife", "party", "clubs", "bars", "live music", "entertainment"}
    user_wants = intent.travel_style == TravelStyle.NIGHTLIFE or bool(
        nightlife_interests & {i.lower() for i in intent.interests}
    )

    if not user_wants:
        return 0.5  # Neutral

    return 1.0 if hotel.near_nightlife else 0.15


def business_match(hotel: Hotel, intent: TripIntent) -> float:
    """Score business amenities for business travellers."""
    from app.schemas.domain import TravelStyle

    user_is_business = intent.travel_style == TravelStyle.BUSINESS

    if not user_is_business:
        return 0.5  # Neutral

    return 1.0 if hotel.business_friendly else 0.2


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

FEATURE_FUNCTIONS = {
    "budget_match": budget_match,
    "location_score": location_score_feature,
    "review_score": review_score_feature,
    "transport_match": transport_match,
    "museum_affinity": museum_affinity,
    "romantic_match": romantic_match,
    "family_match": family_match,
    "nightlife_match": nightlife_match,
    "business_match": business_match,
}


def extract_features(hotel: Hotel, intent: TripIntent) -> FeatureVector:
    """Compute the full feature vector for a (hotel, intent) pair."""
    return FeatureVector(
        hotel_id=hotel.hotel_id,
        budget_match=budget_match(hotel, intent),
        location_score=location_score_feature(hotel, intent),
        review_score=review_score_feature(hotel, intent),
        transport_match=transport_match(hotel, intent),
        museum_affinity=museum_affinity(hotel, intent),
        romantic_match=romantic_match(hotel, intent),
        family_match=family_match(hotel, intent),
        nightlife_match=nightlife_match(hotel, intent),
        business_match=business_match(hotel, intent),
    )
