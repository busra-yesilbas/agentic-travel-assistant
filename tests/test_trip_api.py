"""Tests for POST /trip/plan endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


AMSTERDAM_REQUEST = {
    "query": "I want a 4-day Amsterdam trip for a couple, mid-range budget, close to museums.",
    "use_llm": True,
}

PARIS_REQUEST = {
    "query": "Paris for 3 days, romantic, luxury budget.",
    "city": "Paris",
    "days": 3,
    "budget_level": "luxury",
    "travelers": 2,
    "use_llm": True,
}

BARCELONA_REQUEST = {
    "query": "Barcelona solo 3 days budget traveller, nightlife and architecture.",
    "city": "Barcelona",
    "budget_level": "budget",
    "travelers": 1,
    "use_llm": True,
}


def test_trip_plan_returns_200(client: TestClient) -> None:
    response = client.post("/trip/plan", json=AMSTERDAM_REQUEST)
    assert response.status_code == 200


def test_trip_plan_response_has_required_fields(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    required_fields = [
        "request_id",
        "parsed_intent",
        "ranked_hotels",
        "itinerary",
        "final_answer",
        "critique",
        "stage_latencies",
        "total_latency_ms",
        "metadata",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_trip_plan_returns_hotels(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    assert len(data["ranked_hotels"]) > 0


def test_trip_plan_hotel_structure(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    hotel = data["ranked_hotels"][0]
    assert "hotel" in hotel
    assert "rank" in hotel
    assert "score" in hotel
    assert "explanation" in hotel
    assert hotel["rank"] == 1
    assert 0 <= hotel["score"] <= 1


def test_trip_plan_hotels_ranked_by_score(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    scores = [rh["score"] for rh in data["ranked_hotels"]]
    assert scores == sorted(scores, reverse=True), "Hotels are not sorted by score descending"


def test_trip_plan_correct_city(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    assert data["parsed_intent"]["city"] == "Amsterdam"


def test_trip_plan_city_override(client: TestClient) -> None:
    data = client.post("/trip/plan", json=PARIS_REQUEST).json()
    assert data["parsed_intent"]["city"] == "Paris"


def test_trip_plan_days_override(client: TestClient) -> None:
    data = client.post("/trip/plan", json=PARIS_REQUEST).json()
    assert data["parsed_intent"]["days"] == 3


def test_trip_plan_has_itinerary_days(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    itinerary = data["itinerary"]
    assert "days" in itinerary
    assert len(itinerary["days"]) > 0


def test_trip_plan_itinerary_day_structure(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    day = data["itinerary"]["days"][0]
    assert "day" in day
    assert day["day"] == 1


def test_trip_plan_has_final_answer(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    assert len(data["final_answer"]) > 50


def test_trip_plan_has_critique(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    critique = data["critique"]
    assert "overall_score" in critique
    assert 0 <= critique["overall_score"] <= 1


def test_trip_plan_has_latency(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    assert data["total_latency_ms"] > 0
    assert len(data["stage_latencies"]) > 0


def test_trip_plan_barcelona_budget(client: TestClient) -> None:
    data = client.post("/trip/plan", json=BARCELONA_REQUEST).json()
    assert data["parsed_intent"]["budget_level"] == "budget"
    assert data["parsed_intent"]["city"] == "Barcelona"


def test_trip_plan_invalid_short_query(client: TestClient) -> None:
    response = client.post("/trip/plan", json={"query": "Paris"})
    assert response.status_code == 422


def test_trip_plan_invalid_budget_level(client: TestClient) -> None:
    response = client.post(
        "/trip/plan",
        json={"query": "A trip to Amsterdam please", "budget_level": "super_luxury"},
    )
    assert response.status_code == 422


def test_trip_plan_feature_contributions_present(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    hotel = data["ranked_hotels"][0]
    assert "feature_contributions" in hotel
    assert len(hotel["feature_contributions"]) > 0


def test_trip_plan_hotels_in_requested_city(client: TestClient) -> None:
    data = client.post("/trip/plan", json=AMSTERDAM_REQUEST).json()
    for rh in data["ranked_hotels"]:
        assert rh["hotel"]["city"] == "Amsterdam"
