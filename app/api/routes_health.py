"""GET /health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.metrics import get_registry
from app.schemas.responses import HealthResponse
from app.utils.dates import now_utc_iso

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the service status, version, and runtime configuration.",
)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.tripgenie_env,
        llm_provider=settings.llm_provider,
        timestamp=now_utc_iso(),
    )


@router.get(
    "/metrics",
    summary="Internal metrics snapshot",
    description="Returns in-process counters and latency histograms.",
)
def metrics_snapshot() -> dict:
    return get_registry().snapshot()
