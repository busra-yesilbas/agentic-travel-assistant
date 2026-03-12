# =============================================================================
# TripGenie – Multi-stage Dockerfile
# =============================================================================
FROM python:3.11-slim AS base

LABEL maintainer="TripGenie Team"
LABEL description="TripGenie Agentic Travel Planning Assistant"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Builder stage – install dependencies
# ---------------------------------------------------------------------------
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Create non-root user
RUN groupadd -r tripgenie && useradd -r -g tripgenie tripgenie && \
    chown -R tripgenie:tripgenie /app

USER tripgenie

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
