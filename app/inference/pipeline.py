"""Full inference pipeline for MedVision AI.

Orchestrates:
1. Image Quality Assessment
2. Image Preprocessing
3. Vision Model Inference (DenseNet121)
4. Grad-CAM Attention Heatmap Generation
5. Clinical Context Integration
6. Evidence Grounding
7. Confidence Calibration
8. Explanation Generation
"""

import base64
import uuid
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from app.api.schemas import (
    AnalysisResponse,
    ClinicalContextInput,
    FindingEvidence,
    QualityAssessment,
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.explainability.gradcam import GradCAM
from app.inference.preprocessing import load_image_from_bytes, preprocess_chest_xray
from app.inference.quality import assess_image_quality
from app.models.vision_model import ChestXrayVisionModel, TARGET_PATHOLOGIES
from app.multimodal.clinical_features import extract_clinical_evidence
from app.reasoning.evidence import EvidenceEngine
from app.reasoning.confidence import apply_temperature_scaling
from app.reasoning.safety import format_safety_summary

logger = get_logger("medvision.inference.pipeline")

# Minimum probability for a pathology to enter the reporting pipeline.
# Raised from 0.35 to 0.50 to avoid flooding results when the model
# output is dominated by ImageNet priors (untrained on chest X-ray labels).
REPORTING_THRESHOLD = 0.50

# Probability at which image evidence is considered "strong" for a finding.
# Must be >= REPORTING_THRESHOLD.  Below this the finding is still surfaced
# (if clinical evidence corroborates) but image_evidence is set to False so
# the EvidenceEngine marks it as 'conflicting' rather than 'supported'.
STRONG_EVIDENCE_THRESHOLD = 0.60

# Per-class optional overrides (e.g., to lower threshold for rare but
# clinically critical pathologies). Falls back to REPORTING_THRESHOLD.
CLASS_REPORTING_THRESHOLDS: dict = {
    "Pulmonary_Opacity": 0.50,
    "Pleural_Effusion": 0.50,
    "Cardiomegaly": 0.50,
    "Atelectasis": 0.52,
    "Normal": 0.55,  # require higher confidence to call "Normal"
}

# Clinician-readable pathology names
PATHOLOGY_DISPLAY_NAMES = {
    "Pulmonary_Opacity": "Possible Pulmonary Opacity / Pneumonia",
    "Pleural_Effusion": "Possible Pleural Effusion",
    "Cardiomegaly": "Possible Cardiomegaly",
    "Atelectasis": "Possible Atelectasis",
    "Normal": "No Acute Abnormality",
}


class InferencePipeline:
    """
    End-to-end MedVision AI inference pipeline.

    Thread-safety: vision model and Grad-CAM use torch.no_grad() during inference.
    Model is instantiated once and reused across requests.
    """

    _instance: Optional["InferencePipeline"] = None

    def __init__(self):
        settings = get_settings()
        self.evidence_engine = EvidenceEngine()
        self.device = settings.DEVICE
        checkpoint = str(settings.CHECKPOINT_PATH)

        logger.info("Loading ChestXrayVisionModel on device=%s, checkpoint=%s", self.device, checkpoint)
        self.vision_model = ChestXrayVisionModel(
            checkpoint_path=checkpoint,
            device=self.device,
        )
        self.gradcam = GradCAM(self.vision_model)
        logger.info(
            "Vision model ready. is_trained=%s", self.vision_model.is_trained
        )

    @classmethod
    def get_instance(cls) -> "InferencePipeline":
        """Singleton accessor for the pipeline (avoids reloading model per request)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze(
        self,
        image_bytes: bytes,
        clinical_context: Optional[ClinicalContextInput] = None,
        study_id: Optional[str] = None,
    ) -> AnalysisResponse:
        """
        Run complete multimodal analysis on a submitted chest X-ray.

        Args:
            image_bytes: Raw image file bytes.
            clinical_context: Optional structured clinical context from the doctor.
            study_id: Optional tracking ID. Auto-generated if not provided.

        Returns:
            AnalysisResponse with evidence-grounded findings.
        """
        study_id = study_id or f"CASE-{uuid.uuid4().hex[:8].upper()}"
        logger.info("Starting analysis for study_id=%s", study_id)

        # ── Step 1: Decode Image ──────────────────────────────────────────────
        try:
            image: Image.Image = load_image_from_bytes(image_bytes)
        except ValueError as e:
            raise ValueError(f"Image decoding failed: {e}") from e

        original_size = (image.height, image.width)

        # ── Step 2: Quality Assessment ────────────────────────────────────────
        quality: QualityAssessment = assess_image_quality(image)
        if not quality.is_acceptable:
            logger.warning("study_id=%s quality check FAILED: %s", study_id, quality.issues)

        # ── Step 3: Preprocessing ─────────────────────────────────────────────
        image_tensor = preprocess_chest_xray(image).squeeze(0)  # (C, H, W)

        # ── Step 4: Vision Inference ──────────────────────────────────────────
        raw_probs: Dict[str, float] = self.vision_model.predict(image_tensor)
        logger.info("study_id=%s raw_probs=%s", study_id, raw_probs)

        # ── Step 5: Temperature Scaling Calibration ───────────────────────────
        settings = get_settings()
        calibrated_probs = apply_temperature_scaling(raw_probs, temperature=settings.CALIBRATION_TEMPERATURE)

        # ── Step 6: Clinical Context Evidence ────────────────────────────────
        clinical_evidence_map: Dict[str, bool] = {}
        clinical_concerns: List[str] = []
        missing_fields: List[str] = []
        if clinical_context is not None:
            clin = extract_clinical_evidence(clinical_context)
            clinical_concerns = clin["concerns"]
            missing_fields = clin["missing_fields"]
            for pathology in TARGET_PATHOLOGIES:
                clinical_evidence_map[pathology] = pathology in clin["supported_pathologies"]
        else:
            for pathology in TARGET_PATHOLOGIES:
                clinical_evidence_map[pathology] = False

        # ── Step 7: Grad-CAM Heatmaps ─────────────────────────────────────────
        heatmap_b64_map: Dict[str, str] = {}
        attention_regions: Dict[str, str] = {}

        for class_idx, pathology in enumerate(TARGET_PATHOLOGIES):
            prob = calibrated_probs.get(pathology, 0.0)
            cls_threshold = CLASS_REPORTING_THRESHOLDS.get(pathology, REPORTING_THRESHOLD)
            if prob >= cls_threshold and pathology != "Normal":
                try:
                    heatmap_bytes, region_desc = self.gradcam.generate(
                        image_tensor=image_tensor,
                        class_idx=class_idx,
                        original_size=original_size,
                    )
                    heatmap_b64_map[pathology] = base64.b64encode(heatmap_bytes).decode("utf-8")
                    attention_regions[pathology] = region_desc
                except Exception as e:
                    logger.warning("Grad-CAM failed for %s: %s", pathology, str(e))
                    attention_regions[pathology] = "Attention region could not be computed."

        # ── Step 8: Evidence Grounding ────────────────────────────────────────
        # image_ev is TRUE only when the model probability clears the STRONG
        # evidence bar — not just the reporting floor.  This prevents a finding
        # from being marked 'supported' purely because it crossed the low
        # reporting threshold.
        findings: List[FindingEvidence] = []
        for pathology in TARGET_PATHOLOGIES:
            prob = calibrated_probs.get(pathology, 0.0)
            cls_threshold = CLASS_REPORTING_THRESHOLDS.get(pathology, REPORTING_THRESHOLD)
            if prob < cls_threshold:
                continue

            display_name = PATHOLOGY_DISPLAY_NAMES.get(pathology, pathology)
            # Strong image evidence: probability clearly above the strong threshold.
            image_ev = prob >= STRONG_EVIDENCE_THRESHOLD
            clinical_ev = clinical_evidence_map.get(pathology, False)
            region = attention_regions.get(pathology)

            logger.debug(
                "study_id=%s pathology=%s prob=%.3f image_ev=%s clinical_ev=%s",
                study_id, pathology, prob, image_ev, clinical_ev,
            )

            finding = self.evidence_engine.validate_finding(
                finding_name=display_name,
                probability=prob,
                image_evidence=image_ev,
                clinical_evidence=clinical_ev,
                attention_region=region,
            )
            findings.append(finding)

        # Filter to only assertable findings
        assertable_findings = self.evidence_engine.filter_assertable_findings(findings)

        # ── Step 9: Summary & Recommendations ────────────────────────────────
        summary, recommendations = format_safety_summary(
            findings=assertable_findings,
            quality=quality,
            clinical_concerns=clinical_concerns,
            missing_fields=missing_fields,
            model_is_trained=self.vision_model.is_trained,
        )

        # Attach heatmaps to findings (via model_attention_region field already set)
        # Store heatmaps in finding's model_attention_region for API consumers
        # (Full heatmap bytes are returned in extended response via endpoint)

        logger.info(
            "study_id=%s complete. findings=%d, assertable=%d",
            study_id,
            len(findings),
            len(assertable_findings),
        )

        return AnalysisResponse(
            study_id=study_id,
            quality=quality,
            findings=assertable_findings,
            summary=summary,
            recommendations=recommendations,
        ), heatmap_b64_map
