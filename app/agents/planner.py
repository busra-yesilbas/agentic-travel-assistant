"""
Trip planner entry point.

Provides a high-level async function that the API layer calls.
Converts the raw workflow state into the API response model.
"""

from __future__ import annotations

from app.agents.state import TripPlanningState
from app.agents.workflow import TripPlanningWorkflow
from app.core.exceptions import WorkflowError
from app.core.logging import get_logger
from app.schemas.domain import Critique, Itinerary, TripIntent
from app.schemas.requests import TripPlanningRequest
from app.schemas.responses import TripPlanningResponse

logger = get_logger(__name__)

# Module-level workflow singleton to avoid re-instantiating on every request
_workflow: TripPlanningWorkflow | None = None


def get_workflow() -> TripPlanningWorkflow:
    """Return the singleton workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = TripPlanningWorkflow()
    return _workflow


async def plan_trip(request: TripPlanningRequest) -> TripPlanningResponse:
    """
    Execute the full trip planning pipeline and return an API response.

    Args:
        request: Validated trip planning request.

    Returns:
        TripPlanningResponse with all planning outputs.

    Raises:
        WorkflowError: If the pipeline fails to produce a minimum viable result.
    """
    workflow = get_workflow()
    state = await workflow.run(request)

    if not _has_minimum_output(state):
        raise WorkflowError(
            "Trip planning workflow did not produce a complete result.",
            detail=f"Errors: {state.stage_errors}",
        )

    return _state_to_response(state)


def _has_minimum_output(state: TripPlanningState) -> bool:
    """Check that the state contains enough to build a valid response."""
    return state.parsed_intent is not None and state.final_answer is not None


def _state_to_response(state: TripPlanningState) -> TripPlanningResponse:
    """Map workflow state to the API response model."""
    # Provide safe defaults if optional stages didn't complete
    parsed_intent: TripIntent = state.parsed_intent  # type: ignore[assignment]

    itinerary = state.itinerary or Itinerary(
        trip_name=f"{parsed_intent.city} Trip",
        city=parsed_intent.city,
        total_days=parsed_intent.days,
    )

    critique = state.critique or Critique()

    return TripPlanningResponse(
        request_id=state.request_id,
        parsed_intent=parsed_intent,
        ranked_hotels=state.ranked_hotels,
        itinerary=itinerary,
        final_answer=state.final_answer or "Trip planning complete. See itinerary above.",
        critique=critique,
        stage_latencies=state.stage_latencies,
        total_latency_ms=state.total_latency_ms,
        metadata=state.metadata,
    )
