"""Analysis and inference routes for MedVision AI API."""

import json
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.schemas import AnalysisResponse, ClinicalContextInput, QualityAssessment
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.security import validate_image_upload
from app.inference.pipeline import InferencePipeline

logger = get_logger("medvision.api.routes.analysis")
router = APIRouter(tags=["Analysis"])


def _get_pipeline() -> InferencePipeline:
    """Singleton pipeline accessor — model loads once, reused per request."""
    return InferencePipeline.get_instance()


@router.get(
    "/model/info",
    summary="Model Architecture & Status",
    description="Returns current model architecture status, training state, and clinical scope.",
)
async def get_model_info():
    """Return model metadata without exposing sensitive internals."""
    settings = get_settings()
    try:
        pipeline = _get_pipeline()
        is_trained = pipeline.vision_model.is_trained
        model_status = "Finetuned checkpoint loaded" if is_trained else (
            "Not yet trained on chest X-ray (Pretrained ImageNet backbone only)"
        )
    except Exception as e:
        logger.warning("Pipeline not yet loaded: %s", str(e))
        is_trained = False
        model_status = "Not yet trained (Pipeline uninitialized)"

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "vision_architecture": "DenseNet121 (torchvision pretrained ImageNet backbone)",
        "clinical_task": "Chest X-ray Multi-label Abnormality Detection",
        "target_classes": [
            "Pulmonary_Opacity", "Pleural_Effusion", "Cardiomegaly", "Atelectasis", "Normal"
        ],
        "model_trained": is_trained,
        "model_status": model_status,
        "calibration": "Temperature scaling (configurable via CALIBRATION_TEMPERATURE)",
        "explainability": "Grad-CAM (DenseNet121 final dense block)",
        "device": settings.DEVICE,
        "safety_principle": "Prediction + Localization + Evidence + Confidence",
        "metrics": "Not yet evaluated",
    }


@router.post(
    "/quality-check",
    response_model=QualityAssessment,
    summary="Image Quality Assessment",
    description="Validates image format and assesses diagnostic quality before analysis.",
)
async def quality_check(file: UploadFile = File(...)):
    """Assess image quality without running full inference."""
    from app.inference.preprocessing import load_image_from_bytes
    from app.inference.quality import assess_image_quality

    contents = await file.read()
    valid, message = validate_image_upload(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(contents),
    )
    if not valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    try:
        image = load_image_from_bytes(contents)
        quality = assess_image_quality(image)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image could not be decoded: {str(e)}",
        )

    return quality


@router.post(
    "/analyze",
    summary="Full Multimodal Analysis",
    description=(
        "Runs the complete MedVision AI pipeline: quality check → DenseNet121 inference → "
        "Grad-CAM localization → clinical context integration → evidence grounding → "
        "calibrated confidence → clinician summary. "
        "Returns evidence-grounded findings with attention heatmap data."
    ),
)
async def analyze(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG, max 10MB)"),
    clinical_context: Optional[str] = Form(
        default=None,
        description="JSON-encoded ClinicalContextInput (optional). Provide for multimodal analysis.",
    ),
    study_id: Optional[str] = Form(
        default=None,
        description="Optional de-identified study tracking ID.",
    ),
):
    """
    Full multimodal chest X-ray analysis endpoint.

    Accepts an image upload and optional clinical context JSON.
    Returns evidence-grounded findings, Grad-CAM heatmaps (base64 PNG), and safety summary.

    SAFETY: This endpoint returns AI decision-support output only.
    It does NOT produce a medical diagnosis.
    """
    # Validate upload
    contents = await file.read()
    valid, message = validate_image_upload(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        file_size_bytes=len(contents),
    )
    if not valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    # Parse clinical context
    parsed_context: Optional[ClinicalContextInput] = None
    if clinical_context:
        try:
            context_dict = json.loads(clinical_context)
            parsed_context = ClinicalContextInput(**context_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid clinical_context JSON: {str(e)}",
            )

    # Run pipeline
    try:
        pipeline = _get_pipeline()
        analysis_response, heatmap_b64_map = pipeline.analyze(
            image_bytes=contents,
            clinical_context=parsed_context,
            study_id=study_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Analysis pipeline error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline encountered an internal error.",
        )

    # Return combined response including heatmaps
    response_data = analysis_response.model_dump(mode="json")
    response_data["heatmaps"] = heatmap_b64_map  # {pathology_name: base64_png_string}

    return JSONResponse(content=response_data)
