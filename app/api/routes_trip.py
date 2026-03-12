"""POST /trip/plan endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.agents.planner import plan_trip
from app.core.exceptions import (
    CityNotSupportedError,
    LLMError,
    TripGenieError,
    WorkflowError,
)
from app.core.logging import get_logger
from app.schemas.requests import TripPlanningRequest
from app.schemas.responses import TripPlanningResponse

router = APIRouter(prefix="/trip", tags=["trip"])
logger = get_logger(__name__)


@router.post(
    "/plan",
    response_model=TripPlanningResponse,
    status_code=status.HTTP_200_OK,
    summary="Plan a trip",
    description=(
        "Submit a natural language travel request and receive hotel recommendations, "
        "a day-by-day itinerary, quality critique, and a synthesised final answer."
    ),
)
async def trip_plan(
    request_body: TripPlanningRequest,
    http_request: Request,
) -> TripPlanningResponse:
    """
    Main trip planning endpoint.

    Runs the full agent pipeline: intent extraction → retrieval →
    ranking → itinerary generation → critique → final answer synthesis.
    """
    logger.info(
        "api.trip.plan",
        query_preview=request_body.query[:80],
        city=request_body.city,
        use_llm=request_body.use_llm,
    )

    try:
        response = await plan_trip(request_body)
        return response

    except CityNotSupportedError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": exc.message, "detail": exc.detail},
        ) from exc

    except WorkflowError as exc:
        logger.error("api.trip.workflow_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Trip planning workflow failed.", "detail": exc.detail},
        ) from exc

    except LLMError as exc:
        logger.error("api.trip.llm_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "LLM service unavailable.", "detail": str(exc)},
        ) from exc

    except TripGenieError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.message, "detail": exc.detail},
        ) from exc

    except Exception as exc:
        logger.error("api.trip.unexpected_error", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred.", "detail": str(exc)},
        ) from exc
