"""POST /eval/run endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.schemas.requests import EvalRunRequest
from app.schemas.responses import EvalRunResponse
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/eval", tags=["evaluation"])
logger = get_logger(__name__)


@router.post(
    "/run",
    response_model=EvalRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run offline evaluation",
    description=(
        "Execute the TripGenie pipeline against a set of sample queries "
        "and return aggregate quality metrics."
    ),
)
async def eval_run(request_body: EvalRunRequest) -> EvalRunResponse:
    """
    Offline evaluation endpoint.

    Runs the full planning pipeline for each sample query and returns
    aggregated metrics covering constraint satisfaction, recommendation
    relevance, itinerary completeness, and helpfulness.
    """
    logger.info(
        "api.eval.run",
        num_samples=request_body.num_samples,
        save_results=request_body.save_results,
    )

    try:
        service = ExperimentService()
        response = await service.run_evaluation(
            num_samples=request_body.num_samples,
            save_results=request_body.save_results,
        )

        if not request_body.verbose:
            # Strip per-query details to keep response lightweight
            response = response.model_copy(update={"results": []})

        return response

    except Exception as exc:
        logger.error("api.eval.failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Evaluation run failed.", "detail": str(exc)},
        ) from exc
