"""End-to-end API route tests for MedVision AI."""

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app

client = TestClient(app)


def _create_sample_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create in-memory JPEG bytes for testing — 224×224 with slight noise for non-zero contrast."""
    import numpy as np
    rng = np.random.default_rng(42)
    # Add slight noise around mid-gray so contrast > 0 and sharpness > 0
    pixel_data = rng.integers(100, 160, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixel_data, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_api_health():
    """Verify health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "MedVision AI" in data["project"]


def test_api_quality_check_valid_image():
    """Verify /api/v1/quality-check returns 200 and QualityAssessment schema."""
    jpeg_bytes = _create_sample_jpeg_bytes(256, 256)
    response = client.post(
        "/api/v1/quality-check",
        files={"file": ("chest.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_acceptable" in data
    assert "dimensions" in data
    assert "blur_score" in data


def test_api_quality_check_invalid_file_rejected():
    """Verify unsupported file types are rejected with 400 Bad Request."""
    response = client.post(
        "/api/v1/quality-check",
        files={"file": ("data.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_api_analyze_with_clinical_context():
    """Verify full /api/v1/analyze endpoint processes image and context."""
    jpeg_bytes = _create_sample_jpeg_bytes(256, 256)
    clinical_ctx = {
        "age": 65,
        "sex": "Male",
        "oxygen_saturation": 92.0,
        "symptoms": ["cough", "fever"],
        "relevant_history": "Former smoker",
    }

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("study.jpg", jpeg_bytes, "image/jpeg")},
        data={"clinical_context": json.dumps(clinical_ctx)},
    )

    assert response.status_code == 200
    data = response.json()
    assert "study_id" in data
    assert "quality" in data
    assert "findings" in data
    assert "summary" in data
    assert "recommendations" in data
    assert "heatmaps" in data
