"""
Hotel ranking model.

Implements a weighted linear scorer with full feature explainability.
The design mirrors a practical ML ranking system where weights can be
learned from click/conversion data or set by domain experts.

Architecture:
  candidate hotels → feature extraction → weighted scoring → top-k selection
                                              ↓
                              feature contribution breakdown
                                              ↓
                              natural language explanation
"""

from __future__ import annotations

import numpy as np

from app.ml.features import extract_features
from app.schemas.domain import FeatureContribution, Hotel, RankedHotel, TripIntent

# Default weights (sum to ~1.0). These can be overridden via config.
DEFAULT_WEIGHTS: dict[str, float] = {
    "budget_match": 0.25,
    "location_score": 0.20,
    "review_score": 0.18,
    "transport_match": 0.10,
    "museum_affinity": 0.08,
    "romantic_match": 0.07,
    "family_match": 0.05,
    "nightlife_match": 0.04,
    "business_match": 0.03,
}

# Human-readable names for explanation generation
FEATURE_LABELS: dict[str, str] = {
    "budget_match": "budget compatibility",
    "location_score": "location quality",
    "review_score": "guest review score",
    "transport_match": "transport accessibility",
    "museum_affinity": "proximity to museums",
    "romantic_match": "romantic atmosphere",
    "family_match": "family friendliness",
    "nightlife_match": "nightlife proximity",
    "business_match": "business amenities",
}


class HotelRanker:
    """
    Weighted linear ranker for hotel recommendations.

    Each hotel gets a score in [0, 1] computed as a weighted sum of
    normalised feature values. The scorer also produces per-feature
    contribution breakdowns for interpretability.

    In a production system, the weights would be learned from user
    engagement data (clicks, bookings, reviews). Here they are configured
    via the app YAML for easy experimentation.
    """

    def __init__(self, feature_weights: dict[str, float] | None = None) -> None:
        raw_weights = feature_weights or DEFAULT_WEIGHTS
        # Normalise weights to sum to 1.0
        total = sum(raw_weights.values())
        self._weights = {k: v / total for k, v in raw_weights.items()}

    def score(self, hotel: Hotel, intent: TripIntent) -> tuple[float, list[FeatureContribution]]:
        """
        Compute the weighted score for a single (hotel, intent) pair.

        Returns:
            (score, feature_contributions) where score is in [0, 1]
        """
        features = extract_features(hotel, intent)
        feature_dict = features.to_dict()

        contributions = []
        total_score = 0.0

        for feature_name, raw_value in feature_dict.items():
            weight = self._weights.get(feature_name, 0.0)
            contribution = raw_value * weight
            total_score += contribution
            contributions.append(
                FeatureContribution(
                    feature=feature_name,
                    raw_value=round(raw_value, 4),
                    weight=round(weight, 4),
                    contribution=round(contribution, 4),
                )
            )

        # Sort contributions by impact (descending) for explanation
        contributions.sort(key=lambda c: c.contribution, reverse=True)
        return round(float(np.clip(total_score, 0.0, 1.0)), 4), contributions

    def rank(
        self, hotels: list[Hotel], intent: TripIntent, top_k: int = 5
    ) -> list[RankedHotel]:
        """
        Score and rank all candidate hotels.

        Args:
            hotels: Candidate hotels from the retrieval stage.
            intent: Structured user preferences.
            top_k: Number of top hotels to return.

        Returns:
            List of RankedHotel objects sorted by score descending.
        """
        scored = []
        for hotel in hotels:
            score, contributions = self.score(hotel, intent)
            explanation = self._build_explanation(hotel, contributions, score)
            scored.append((score, hotel, contributions, explanation))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        ranked = []
        for rank_idx, (score, hotel, contributions, explanation) in enumerate(
            scored[:top_k], start=1
        ):
            ranked.append(
                RankedHotel(
                    hotel=hotel,
                    rank=rank_idx,
                    score=score,
                    feature_contributions=contributions,
                    explanation=explanation,
                )
            )

        return ranked

    @staticmethod
    def _build_explanation(
        hotel: Hotel,
        contributions: list[FeatureContribution],
        score: float,
    ) -> str:
        """Generate a concise natural language explanation for a hotel's ranking."""
        # Top 3 contributing features (excluding neutral 0.5-weight features)
        top_features = [
            c for c in contributions[:4] if c.raw_value >= 0.6
        ][:3]

        if not top_features:
            return f"{hotel.name} scores {score:.0%} overall with solid baseline ratings."

        reasons = [FEATURE_LABELS.get(c.feature, c.feature) for c in top_features]

        if len(reasons) == 1:
            reason_str = reasons[0]
        elif len(reasons) == 2:
            reason_str = f"{reasons[0]} and {reasons[1]}"
        else:
            reason_str = f"{reasons[0]}, {reasons[1]}, and {reasons[2]}"

        score_pct = f"{score:.0%}"
        return (
            f"Ranked #{hotel.hotel_id[-3:]} with {score_pct} match. "
            f"Strongest signals: {reason_str}."
        )
