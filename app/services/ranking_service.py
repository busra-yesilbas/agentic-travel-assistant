"""
Ranking service.

Orchestrates the ML-based hotel ranking pipeline: feature engineering,
weighted scoring, and explainability output.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.ranker import HotelRanker
from app.schemas.domain import Hotel, RankedHotel, TripIntent

logger = get_logger(__name__)


class RankingService:
    """
    Scores and ranks candidate hotels for a given trip intent.

    Wraps the ML ranker and adds service-level concerns like
    logging and top-k selection.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._ranker = HotelRanker(
            feature_weights=self._settings.ranking_feature_weights
        )

    def rank_hotels(
        self,
        hotels: list[Hotel],
        intent: TripIntent,
        top_k: int | None = None,
    ) -> list[RankedHotel]:
        """
        Rank hotels by relevance to the trip intent.

        Args:
            hotels: Candidate hotels from the retrieval stage.
            intent: Structured user trip preferences.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of hotels with scores and explanations.
        """
        if not hotels:
            logger.warning("ranking.empty_candidate_set")
            return []

        k = top_k or self._settings.ranking_top_k
        ranked = self._ranker.rank(hotels, intent, top_k=k)

        logger.info(
            "ranking.complete",
            candidates=len(hotels),
            returned=len(ranked),
            top_score=ranked[0].score if ranked else 0.0,
            city=intent.city,
        )
        return ranked
