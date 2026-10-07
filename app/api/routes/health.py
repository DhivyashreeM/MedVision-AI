"""Health check and system status endpoints."""

from fastapi import APIRouter, status
from app.api.schemas import HealthResponse
from app.core.config import get_settings

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the runtime health, service version, and model/dataset integration status.",
)
async def health_check() -> HealthResponse:
    """Return structured health status of the MedVision AI platform."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        model_status="Not yet integrated (Phase 1 foundation)",
        dataset_status="Not yet integrated (Phase 1 foundation)",
        safety_notice="AI-assisted decision support only. This system does not replace professional medical judgment.",
    )
