"""
Dataset service.

Loads and caches the static CSV and JSON datasets from the data/ directory.
All data loading happens lazily on first access and is cached for the
application lifetime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import DataLoadError
from app.core.logging import get_logger
from app.schemas.domain import (
    SUPPORTED_CITIES,
    Attraction,
    CityGuide,
    Hotel,
    Restaurant,
)

logger = get_logger(__name__)


class DatasetService:
    """
    Loads and serves the travel datasets.

    Provides typed, validated domain objects rather than raw DataFrames
    to the rest of the application.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._hotels: list[Hotel] | None = None
        self._attractions: list[Attraction] | None = None
        self._restaurants: list[Restaurant] | None = None
        self._city_guides: dict[str, CityGuide] | None = None

    # ------------------------------------------------------------------
    # Hotels
    # ------------------------------------------------------------------
    def get_hotels(self, city: str | None = None) -> list[Hotel]:
        """Return all hotels, optionally filtered by city."""
        if self._hotels is None:
            self._hotels = self._load_hotels()
        if city:
            return [h for h in self._hotels if h.city.lower() == city.lower()]
        return self._hotels

    def _load_hotels(self) -> list[Hotel]:
        path = self._settings.hotels_path
        df = self._read_csv(path, "hotels")

        bool_cols = [
            "family_friendly", "business_friendly", "near_museum",
            "near_nightlife", "near_public_transport", "romantic", "luxury", "budget",
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int).astype(bool)

        hotels = []
        for _, row in df.iterrows():
            try:
                hotel = Hotel(
                    hotel_id=str(row["hotel_id"]),
                    city=str(row["city"]),
                    name=str(row["name"]),
                    price_level=int(row["price_level"]),
                    avg_review_score=float(row["avg_review_score"]),
                    location_score=float(row["location_score"]),
                    family_friendly=bool(row.get("family_friendly", False)),
                    business_friendly=bool(row.get("business_friendly", False)),
                    near_museum=bool(row.get("near_museum", False)),
                    near_nightlife=bool(row.get("near_nightlife", False)),
                    near_public_transport=bool(row.get("near_public_transport", True)),
                    romantic=bool(row.get("romantic", False)),
                    luxury=bool(row.get("luxury", False)),
                    budget=bool(row.get("budget", False)),
                    description=str(row.get("description", "")),
                )
                hotels.append(hotel)
            except Exception as exc:
                logger.warning("dataset.hotel_parse_error", row_id=row.get("hotel_id"), error=str(exc))

        logger.info("dataset.hotels_loaded", count=len(hotels))
        return hotels

    # ------------------------------------------------------------------
    # Attractions
    # ------------------------------------------------------------------
    def get_attractions(self, city: str | None = None) -> list[Attraction]:
        """Return all attractions, optionally filtered by city."""
        if self._attractions is None:
            self._attractions = self._load_attractions()
        if city:
            return [a for a in self._attractions if a.city.lower() == city.lower()]
        return self._attractions

    def _load_attractions(self) -> list[Attraction]:
        path = self._settings.attractions_path
        df = self._read_csv(path, "attractions")

        attractions = []
        for _, row in df.iterrows():
            try:
                tags_raw = str(row.get("tags", ""))
                # Tags may be stored as a quoted comma-separated string
                tags = [t.strip().strip('"\'[]') for t in tags_raw.split(",") if t.strip()]
                attraction = Attraction(
                    attraction_id=str(row["attraction_id"]),
                    city=str(row["city"]),
                    name=str(row["name"]),
                    category=str(row.get("category", "sightseeing")),
                    tags=tags,
                    duration_hours=float(row.get("duration_hours", 2.0)),
                    price_level=int(row.get("price_level", 1)),
                    rating=float(row.get("rating", 8.0)),
                    description=str(row.get("description", "")),
                )
                attractions.append(attraction)
            except Exception as exc:
                logger.warning(
                    "dataset.attraction_parse_error",
                    row_id=row.get("attraction_id"),
                    error=str(exc),
                )

        logger.info("dataset.attractions_loaded", count=len(attractions))
        return attractions

    # ------------------------------------------------------------------
    # Restaurants
    # ------------------------------------------------------------------
    def get_restaurants(self, city: str | None = None) -> list[Restaurant]:
        """Return all restaurants, optionally filtered by city."""
        if self._restaurants is None:
            self._restaurants = self._load_restaurants()
        if city:
            return [r for r in self._restaurants if r.city.lower() == city.lower()]
        return self._restaurants

    def _load_restaurants(self) -> list[Restaurant]:
        path = self._settings.restaurants_path
        df = self._read_csv(path, "restaurants")

        bool_cols = ["near_transport", "romantic", "family_friendly", "vegetarian_friendly"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int).astype(bool)

        restaurants = []
        for _, row in df.iterrows():
            try:
                rest = Restaurant(
                    restaurant_id=str(row["restaurant_id"]),
                    city=str(row["city"]),
                    name=str(row["name"]),
                    cuisine=str(row.get("cuisine", "International")),
                    price_level=int(row.get("price_level", 2)),
                    rating=float(row.get("rating", 8.0)),
                    near_transport=bool(row.get("near_transport", True)),
                    romantic=bool(row.get("romantic", False)),
                    family_friendly=bool(row.get("family_friendly", False)),
                    vegetarian_friendly=bool(row.get("vegetarian_friendly", False)),
                    description=str(row.get("description", "")),
                )
                restaurants.append(rest)
            except Exception as exc:
                logger.warning(
                    "dataset.restaurant_parse_error",
                    row_id=row.get("restaurant_id"),
                    error=str(exc),
                )

        logger.info("dataset.restaurants_loaded", count=len(restaurants))
        return restaurants

    # ------------------------------------------------------------------
    # City guides
    # ------------------------------------------------------------------
    def get_city_guide(self, city: str) -> CityGuide | None:
        """Return the city guide for the given city, or None if not found."""
        if self._city_guides is None:
            self._city_guides = self._load_city_guides()
        return self._city_guides.get(city)

    def _load_city_guides(self) -> dict[str, CityGuide]:
        path = self._settings.city_guides_path
        if not path.exists():
            logger.warning("dataset.city_guides_not_found", path=str(path))
            return {}
        try:
            with path.open() as fh:
                raw = json.load(fh)
            guides = {}
            for city, data in raw.items():
                try:
                    guides[city] = CityGuide(**data)
                except Exception as exc:
                    logger.warning("dataset.city_guide_parse_error", city=city, error=str(exc))
            logger.info("dataset.city_guides_loaded", count=len(guides))
            return guides
        except Exception as exc:
            logger.error("dataset.city_guides_load_failed", error=str(exc))
            return {}

    def get_supported_cities(self) -> list[str]:
        """Return the list of cities available in the dataset."""
        if self._hotels is None:
            self._hotels = self._load_hotels()
        cities = sorted({h.city for h in self._hotels})
        return cities or SUPPORTED_CITIES

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _read_csv(self, path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise DataLoadError(
                f"Dataset file not found: {path}",
                detail=f"Expected {name} data at {path}. Run the data generation script.",
            )
        try:
            df = pd.read_csv(path)
            logger.debug("dataset.csv_loaded", name=name, rows=len(df), path=str(path))
            return df
        except Exception as exc:
            raise DataLoadError(f"Failed to load {name} dataset: {exc}") from exc


@lru_cache(maxsize=1)
def get_dataset_service() -> DatasetService:
    """Return the singleton DatasetService instance."""
    return DatasetService()
