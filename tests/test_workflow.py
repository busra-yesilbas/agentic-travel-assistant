"""Tests for the agent workflow pipeline."""

from __future__ import annotations

import pytest

from app.agents.workflow import TripPlanningWorkflow
from app.schemas.requests import TripPlanningRequest


@pytest.fixture(scope="module")
def workflow() -> TripPlanningWorkflow:
    return TripPlanningWorkflow(force_mock=True)


@pytest.mark.asyncio
async def test_workflow_completes_successfully(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="4-day Amsterdam trip for a couple, mid-range budget, museums and canals.",
        use_llm=True,
    )
    state = await workflow.run(request)
    assert state.is_complete


@pytest.mark.asyncio
async def test_workflow_populates_intent(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="3-day Paris trip, romantic, luxury budget.",
        city="Paris",
        days=3,
        use_llm=True,
    )
    state = await workflow.run(request)
    assert state.parsed_intent is not None
    assert state.parsed_intent.city == "Paris"
    assert state.parsed_intent.days == 3


@pytest.mark.asyncio
async def test_workflow_produces_ranked_hotels(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Amsterdam 3 days mid-range budget.",
    )
    state = await workflow.run(request)
    assert len(state.ranked_hotels) > 0
    assert state.ranked_hotels[0].rank == 1


@pytest.mark.asyncio
async def test_workflow_produces_itinerary(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Barcelona 3 days, culture and food.",
        city="Barcelona",
    )
    state = await workflow.run(request)
    assert state.itinerary is not None
    assert state.itinerary.city == "Barcelona"
    assert len(state.itinerary.days) > 0


@pytest.mark.asyncio
async def test_workflow_produces_critique(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Rome 4 days, history and culture, mid-range.",
        city="Rome",
    )
    state = await workflow.run(request)
    assert state.critique is not None
    assert 0 <= state.critique.overall_score <= 1


@pytest.mark.asyncio
async def test_workflow_produces_final_answer(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Lisbon 4 days, fado and food, mid-range.",
        city="Lisbon",
    )
    state = await workflow.run(request)
    assert state.final_answer is not None
    assert len(state.final_answer) > 50


@pytest.mark.asyncio
async def test_workflow_records_all_stage_latencies(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Amsterdam 4 days, museums, mid-range.",
    )
    state = await workflow.run(request)
    expected_stages = {"intent", "retrieval", "ranking", "itinerary", "critique", "answer"}
    recorded_stages = set(state.stage_latencies.keys())
    assert expected_stages.issubset(recorded_stages)


@pytest.mark.asyncio
async def test_workflow_total_latency_positive(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Paris 5 days family trip.",
    )
    state = await workflow.run(request)
    assert state.total_latency_ms > 0


@pytest.mark.asyncio
async def test_workflow_request_id_is_set(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Amsterdam 3 days.",
    )
    state = await workflow.run(request)
    assert state.request_id
    assert len(state.request_id) > 0


@pytest.mark.asyncio
async def test_workflow_hotels_in_correct_city(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Barcelona 3 days, architecture and food.",
        city="Barcelona",
    )
    state = await workflow.run(request)
    for rh in state.ranked_hotels:
        assert rh.hotel.city == "Barcelona"


@pytest.mark.asyncio
async def test_workflow_with_interests_override(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="Amsterdam 3 days.",
        interests=["museums", "cycling"],
    )
    state = await workflow.run(request)
    assert state.parsed_intent is not None
    interests_set = {i.lower() for i in state.parsed_intent.interests}
    # The override interests should be merged in
    assert "museums" in interests_set or "cycling" in interests_set


@pytest.mark.asyncio
async def test_workflow_mock_flag_set(workflow: TripPlanningWorkflow) -> None:
    request = TripPlanningRequest(
        query="4-day Amsterdam trip, mid-range.",
    )
    state = await workflow.run(request)
    # In test mode with force_mock=True, mock_llm_used should be True
    assert state.mock_llm_used is True
