"""
Core domain models.

These represent the internal data structures that flow through the agent
pipeline. They are separate from the API request/response schemas to keep
the domain model independent of the transport layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class BudgetLevel(StrEnum):
    BUDGET = "budget"
    MID = "mid"
    UPPER_MID = "upper_mid"
    LUXURY = "luxury"


class TravelStyle(StrEnum):
    CULTURAL = "cultural"
    ADVENTURE = "adventure"
    RELAXATION = "relaxation"
    NIGHTLIFE = "nightlife"
    FAMILY = "family"
    BUSINESS = "business"
    ROMANTIC = "romantic"


SUPPORTED_CITIES = ["Amsterdam", "Paris", "Barcelona", "Rome", "Lisbon"]


# ---------------------------------------------------------------------------
# Trip Intent
# ---------------------------------------------------------------------------
class TripIntent(BaseModel):
    """Structured representation of what the user wants from their trip."""

    city: str = Field(..., description="Primary destination city")
    days: int = Field(default=3, ge=1, le=30, description="Trip duration in days")
    travelers: int = Field(default=2, ge=1, le=20, description="Number of travelers")
    budget_level: BudgetLevel = Field(default=BudgetLevel.MID)
    interests: list[str] = Field(default_factory=list)
    travel_style: TravelStyle = Field(default=TravelStyle.CULTURAL)
    accommodation_preferences: list[str] = Field(default_factory=list)
    special_requests: list[str] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None

    @field_validator("city")
    @classmethod
    def normalise_city(cls, v: str) -> str:
        for city in SUPPORTED_CITIES:
            if city.lower() == v.strip().lower():
                return city
        # Fuzzy match by prefix
        for city in SUPPORTED_CITIES:
            if city.lower().startswith(v.strip().lower()[:3]):
                return city
        return v.strip().title()

    def to_summary(self) -> str:
        parts = [
            f"{self.days}-day trip to {self.city}",
            f"for {self.travelers} traveller(s)",
            f"({self.budget_level.value} budget)",
        ]
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests[:4])}")
        if self.special_requests:
            parts.append(f"Special: {', '.join(self.special_requests[:3])}")
        return ". ".join(parts)


# ---------------------------------------------------------------------------
# Hotel
# ---------------------------------------------------------------------------
class Hotel(BaseModel):
    """A hotel from the dataset."""

    hotel_id: str
    city: str
    name: str
    price_level: int = Field(ge=1, le=4)
    avg_review_score: float = Field(ge=0.0, le=10.0)
    location_score: float = Field(ge=0.0, le=10.0)
    family_friendly: bool = False
    business_friendly: bool = False
    near_museum: bool = False
    near_nightlife: bool = False
    near_public_transport: bool = True
    romantic: bool = False
    luxury: bool = False
    budget: bool = False
    description: str = ""

    @property
    def price_label(self) -> str:
        return {1: "Budget", 2: "Mid-range", 3: "Upper mid-range", 4: "Luxury"}[self.price_level]


# ---------------------------------------------------------------------------
# Ranked Hotel
# ---------------------------------------------------------------------------
class FeatureContribution(BaseModel):
    """Score contribution from one feature."""

    feature: str
    raw_value: float
    weight: float
    contribution: float


class RankedHotel(BaseModel):
    """A hotel with its ranking score and explainability metadata."""

    hotel: Hotel
    rank: int
    score: float = Field(ge=0.0, le=1.0)
    feature_contributions: list[FeatureContribution] = Field(default_factory=list)
    explanation: str = ""

    @property
    def name(self) -> str:
        return self.hotel.name

    @property
    def city(self) -> str:
        return self.hotel.city


# ---------------------------------------------------------------------------
# Attraction & Restaurant
# ---------------------------------------------------------------------------
class Attraction(BaseModel):
    """A point of interest from the dataset."""

    attraction_id: str
    city: str
    name: str
    category: str
    tags: list[str] = Field(default_factory=list)
    duration_hours: float = 2.0
    price_level: int = Field(default=1, ge=0, le=4)
    rating: float = Field(ge=0.0, le=10.0, default=8.0)
    description: str = ""


class Restaurant(BaseModel):
    """A restaurant from the dataset."""

    restaurant_id: str
    city: str
    name: str
    cuisine: str
    price_level: int = Field(ge=1, le=4)
    rating: float = Field(ge=0.0, le=10.0, default=8.0)
    near_transport: bool = True
    romantic: bool = False
    family_friendly: bool = False
    vegetarian_friendly: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Itinerary
# ---------------------------------------------------------------------------
class ItineraryDay(BaseModel):
    """A single day in the trip itinerary."""

    day: int
    theme: str = ""
    morning: str = ""
    afternoon: str = ""
    evening: str = ""
    recommended_hotel: str = ""
    transport_tip: str = ""
    estimated_daily_budget: str = ""


class Itinerary(BaseModel):
    """The complete day-by-day trip plan."""

    trip_name: str
    city: str
    total_days: int
    overview: str = ""
    days: list[ItineraryDay] = Field(default_factory=list)
    total_estimated_cost: str = ""
    practical_tips: list[str] = Field(default_factory=list)
    best_time_to_visit: str = ""

    def to_summary(self) -> str:
        lines = [f"**{self.trip_name}** — {self.total_days} days in {self.city}"]
        if self.overview:
            lines.append(self.overview)
        for day in self.days:
            lines.append(
                f"Day {day.day} ({day.theme}): {day.morning} | {day.afternoon} | {day.evening}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------
class Critique(BaseModel):
    """Quality assessment of the generated travel plan."""

    budget_respected: bool = True
    duration_included: bool = True
    activities_sufficient: bool = True
    hotel_alignment: bool = True
    assumptions_stated: bool = False
    overall_score: float = Field(default=0.8, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    approved: bool = True

    @property
    def quality_label(self) -> str:
        if self.overall_score >= 0.85:
            return "Excellent"
        elif self.overall_score >= 0.70:
            return "Good"
        elif self.overall_score >= 0.55:
            return "Fair"
        return "Needs improvement"


# ---------------------------------------------------------------------------
# City Guide
# ---------------------------------------------------------------------------
class CityGuide(BaseModel):
    """City-level reference information."""

    country: str
    language: str
    currency: str
    timezone: str
    best_months: list[str] = Field(default_factory=list)
    avg_daily_budget: dict[str, str] = Field(default_factory=dict)
    highlights: list[str] = Field(default_factory=list)
    day_trips: list[str] = Field(default_factory=list)
    practical_info: dict[str, Any] = Field(default_factory=dict)
