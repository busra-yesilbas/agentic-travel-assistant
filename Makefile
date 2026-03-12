.PHONY: help install install-dev run run-dev test test-cov lint format typecheck clean docker-build docker-up docker-down ui eval

PYTHON := python
UVICORN := uvicorn
APP_MODULE := app.main:create_app
HOST := 0.0.0.0
PORT := 8000

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install all dependencies including dev tools
	pip install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
run:  ## Start the API server (production mode)
	$(UVICORN) "$(APP_MODULE)" --factory --host $(HOST) --port $(PORT)

run-dev:  ## Start the API server with auto-reload
	$(UVICORN) "$(APP_MODULE)" --factory --host $(HOST) --port $(PORT) --reload --log-level debug

ui:  ## Start the Streamlit demo UI
	streamlit run app/ui/streamlit_app.py --server.port 8501

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:  ## Run the test suite
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=app --cov-report=term-missing --cov-report=html

test-watch:  ## Run tests in watch mode (requires pytest-watch)
	ptw -- -x

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint:  ## Run Ruff linter
	ruff check app tests

format:  ## Format code with Ruff
	ruff format app tests

format-check:  ## Check formatting without modifying files
	ruff format --check app tests

typecheck:  ## Run mypy type checking
	mypy app

check: lint format-check typecheck  ## Run all checks

fix:  ## Auto-fix linting issues
	ruff check --fix app tests
	ruff format app tests

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
eval:  ## Run offline evaluation against sample queries
	$(PYTHON) -c "import asyncio; from app.services.experiment_service import ExperimentService; asyncio.run(ExperimentService().run_evaluation())"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker build -t tripgenie:latest .

docker-up:  ## Start all services with docker-compose
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## Tail logs from docker-compose
	docker compose logs -f api

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:  ## Remove build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
