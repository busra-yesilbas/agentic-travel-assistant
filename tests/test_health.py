"""Tests for GET /health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert "llm_provider" in data
    assert "timestamp" in data


def test_health_version_format(client: TestClient) -> None:
    data = client.get("/health").json()
    parts = data["version"].split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_health_llm_provider_is_mock(client: TestClient) -> None:
    """In the test environment, the LLM provider should be 'mock'."""
    data = client.get("/health").json()
    assert data["llm_provider"] == "mock"


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "counters" in data
    assert "histograms" in data
