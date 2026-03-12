"""
TripGenie FastAPI application factory.

Usage:
    uvicorn app.main:create_app --factory --reload
    # or
    python -m app.main
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_eval import router as eval_router
from app.api.routes_health import router as health_router
from app.api.routes_trip import router as trip_router
from app.core.config import get_settings
from app.core.exceptions import TripGenieError
from app.core.logging import configure_logging, get_logger
from app.core.metrics import get_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    logger = get_logger("tripgenie.startup")

    logger.info(
        "tripgenie.starting",
        version=settings.app_version,
        environment=settings.tripgenie_env,
        llm_provider=settings.llm_provider,
    )

    # Warm up the dataset service on startup
    from app.services.dataset_service import get_dataset_service
    dataset = get_dataset_service()
    hotels = dataset.get_hotels()
    attractions = dataset.get_attractions()
    logger.info(
        "tripgenie.data_loaded",
        hotels=len(hotels),
        attractions=len(attractions),
    )

    # Pre-warm the workflow singleton
    from app.agents.planner import get_workflow
    get_workflow()
    logger.info("tripgenie.ready")

    yield

    logger.info("tripgenie.shutting_down")


def create_app() -> FastAPI:
    """Application factory. Called by Uvicorn when using --factory flag."""
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="TripGenie",
        description=(
            "Agentic Travel Planning & Booking Assistant. "
            "An LLM-powered travel planner with ML-based hotel ranking, "
            "day-by-day itinerary generation, and quality evaluation."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------------------
    # CORS
    # ---------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Request timing middleware
    # ---------------------------------------------------------------------------
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        get_registry().histogram("http.request_latency_ms").observe(elapsed_ms)
        return response

    # ---------------------------------------------------------------------------
    # Global exception handler for TripGenieError
    # ---------------------------------------------------------------------------
    @app.exception_handler(TripGenieError)
    async def tripgenie_exception_handler(
        request: Request, exc: TripGenieError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "detail": exc.detail},
        )

    # ---------------------------------------------------------------------------
    # Routers
    # ---------------------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(trip_router)
    app.include_router(eval_router)

    return app


def main() -> None:
    """Entry point for running the server directly."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.tripgenie_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
