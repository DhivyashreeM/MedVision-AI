"""Pydantic request and response schemas for MedVision AI API."""

from datetime import datetime, timezone
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class HealthResponse(BaseModel):
    """Structured health-check response model."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        default="healthy", description="Current service health status"
    )
    project: str = Field(default="MedVision AI", description="System identifier")
    version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Runtime environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of health check",
    )
    model_status: str = Field(
        default="Not yet integrated (Phase 1 foundation)",
        description="Status of vision and fusion model weights",
    )
    dataset_status: str = Field(
        default="Not yet integrated (Phase 1 foundation)",
        description="Status of clinical training datasets",
    )
    safety_notice: str = Field(
        default="AI-assisted decision support only. This system does not replace professional medical judgment.",
        description="Mandatory medical disclaimer",
    )


class QualityAssessment(BaseModel):
    """Image quality assessment results."""

    is_acceptable: bool = Field(..., description="Whether image meets minimal diagnostic criteria")
    dimensions: List[int] = Field(default_factory=list, description="[Height, Width, Channels]")
    blur_score: Optional[float] = Field(default=None, description="Laplacian variance or sharpness metric")
    contrast_score: Optional[float] = Field(default=None, description="Contrast range score")
    brightness_score: Optional[float] = Field(default=None, description="Mean intensity score")
    issues: List[str] = Field(default_factory=list, description="Identified quality defects")
    warning: Optional[str] = Field(
        default=None, description="Clinician warning if quality is degraded"
    )


class ClinicalContextInput(BaseModel):
    """Clinical context provided alongside imaging study."""

    model_config = ConfigDict(extra="ignore")

    age: Optional[int] = Field(default=None, ge=0, le=125, description="Patient age in years")
    sex: Optional[Literal["male", "female", "other", "unspecified"]] = Field(
        default=None, description="Biological sex"
    )

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex(cls, v):
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ("male", "female", "other", "unspecified"):
                return v_lower
        return v
    symptoms: List[str] = Field(
        default_factory=list, description="Reported symptoms (e.g., cough, fever, dyspnea)"
    )
    oxygen_saturation: Optional[float] = Field(
        default=None, ge=50.0, le=100.0, description="SpO2 percentage"
    )
    smoking_history: Optional[str] = Field(
        default=None, description="Smoking history if provided (do not assume non-smoker if missing)"
    )
    relevant_history: Optional[str] = Field(
        default=None, description="Prior conditions, procedures, or relevant context"
    )
    clinical_notes: Optional[str] = Field(
        default=None, description="De-identified physician or triage notes"
    )


class FindingEvidence(BaseModel):
    """Evidence-grounded finding assertion."""

    finding: str = Field(..., description="Suspected clinical finding (e.g., Pulmonary Opacity)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model calibrated probability score (0.0 to 1.0)"
    )
    confidence_level: Literal["High", "Moderate", "Low"] = Field(
        ..., description="Categorical confidence level"
    )
    image_evidence: bool = Field(
        ..., description="True if visual image attention supports this finding"
    )
    clinical_evidence: bool = Field(
        ..., description="True if supplied clinical context supports this finding"
    )
    evidence_status: Literal["supported", "insufficient", "conflicting", "not_detected"] = Field(
        ..., description="Synthesized evidence grounding determination"
    )
    model_attention_region: Optional[str] = Field(
        default=None,
        description="Anatomical attention description (e.g., Right lower lung field)",
    )
    supporting_rationale: str = Field(
        ..., description="Clear explanation grounded in verified evidence"
    )


class AnalysisResponse(BaseModel):
    """Full decision-support analysis response."""

    study_id: str = Field(..., description="De-identified study tracking identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis completion timestamp",
    )
    quality: QualityAssessment = Field(..., description="Image quality review")
    findings: List[FindingEvidence] = Field(
        default_factory=list, description="Evidence-grounded findings list"
    )
    summary: str = Field(..., description="Clinician-oriented synthesis summary")
    recommendations: List[str] = Field(
        default_factory=list, description="Suggested decision-support review actions"
    )
    safety_disclaimer: str = Field(
        default="AI-assisted decision support only. This system does not replace professional medical judgment.",
        description="Mandatory clinician disclaimer",
    )


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Internal error code")
    details: Optional[str] = Field(default=None, description="User-safe error explanation")
