"""Tests for system health endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.api.main import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


def test_health_check_endpoint(client):
    """Test root /health endpoint returns structured health status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "MedVision AI"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
    assert "Not yet integrated" in data["model_status"]
    assert "Not yet integrated" in data["dataset_status"]
    assert "AI-assisted decision support" in data["safety_notice"]


def test_api_v1_health_endpoint(client):
    """Test /api/v1/health route returns matching payload."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_model_info_endpoint(client):
    """Test /api/v1/model/info reports honest non-fabricated status."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["metrics"] == "Not yet evaluated"
    assert data["model_status"] in ("Finetuned checkpoint loaded", "Not yet trained") or "checkpoint" in data["model_status"].lower()


