"""Clinical context processing for MedVision AI multimodal fusion.

Extracts structured clinical evidence signals from the ClinicalContextInput
to support evidence-grounded finding assessment.

AGENTS.md Rule 10: Never convert missing information into negative information.
"""

from typing import Any, Dict, List, Optional, Set

from app.api.schemas import ClinicalContextInput
from app.core.logging import get_logger

logger = get_logger("medvision.multimodal.clinical")

# Symptom → supported pathology keyword mapping
SYMPTOM_PATHOLOGY_MAP: Dict[str, List[str]] = {
    "cough": ["Pulmonary_Opacity", "Atelectasis"],
    "fever": ["Pulmonary_Opacity"],
    "shortness of breath": ["Pleural_Effusion", "Pulmonary_Opacity", "Cardiomegaly"],
    "dyspnea": ["Pleural_Effusion", "Pulmonary_Opacity", "Cardiomegaly"],
    "chest pain": ["Pulmonary_Opacity", "Pleural_Effusion"],
    "hypoxia": ["Pulmonary_Opacity", "Pleural_Effusion"],
    "wheezing": ["Pulmonary_Opacity"],
    "leg swelling": ["Cardiomegaly", "Pleural_Effusion"],
    "orthopnea": ["Cardiomegaly", "Pleural_Effusion"],
    "hemoptysis": ["Pulmonary_Opacity"],
}


def extract_clinical_evidence(
    context: ClinicalContextInput,
) -> Dict[str, Any]:
    """
    Map clinical context to pathology-level evidence signals.

    Returns:
        {
            "supported_pathologies": set of pathology names supported by clinical context,
            "clinical_features": dict of structured clinical signals,
            "concerns": list of clinical concern strings (for display),
            "missing_fields": list of fields not provided (to avoid false negatives),
        }
    """
    supported: Set[str] = set()
    concerns: List[str] = []
    missing_fields: List[str] = []

    # --- Age ---
    if context.age is None:
        missing_fields.append("age")
    else:
        if context.age >= 65:
            concerns.append(f"Age {context.age}: increased cardiovascular and pulmonary risk.")

    # --- Symptoms ---
    if not context.symptoms:
        missing_fields.append("symptoms")
    else:
        for symptom in context.symptoms:
            sym_lower = symptom.lower().strip()
            for key, pathologies in SYMPTOM_PATHOLOGY_MAP.items():
                if key in sym_lower:
                    supported.update(pathologies)
                    concerns.append(f"Symptom '{symptom}' may be associated with: {', '.join(pathologies)}.")

    # --- O2 Saturation ---
    if context.oxygen_saturation is None:
        missing_fields.append("oxygen_saturation")
    else:
        if context.oxygen_saturation < 94.0:
            supported.update(["Pulmonary_Opacity", "Pleural_Effusion"])
            concerns.append(f"SpO2 {context.oxygen_saturation:.0f}%: below normal threshold — pulmonary concern.")

    # --- Smoking History ---
    if context.smoking_history is None:
        missing_fields.append("smoking_history")
    # NOTE: We do NOT infer 'non-smoker' from absence — AGENTS.md Rule 10

    # --- Relevant History ---
    if context.relevant_history:
        history_lower = context.relevant_history.lower()
        if any(kw in history_lower for kw in ["heart failure", "cardiac", "cardiomegaly"]):
            supported.add("Cardiomegaly")
            concerns.append("History mentions cardiac condition — Cardiomegaly supported by history.")
        if any(kw in history_lower for kw in ["effusion", "pleural"]):
            supported.add("Pleural_Effusion")
        if any(kw in history_lower for kw in ["pneumonia", "opacity", "consolidation", "infection"]):
            supported.add("Pulmonary_Opacity")
        if any(kw in history_lower for kw in ["atelectasis", "collapse"]):
            supported.add("Atelectasis")

    clinical_features = {
        "age": context.age,
        "sex": context.sex,
        "oxygen_saturation": context.oxygen_saturation,
        "symptoms": context.symptoms,
        "smoking_history": context.smoking_history or "Not provided",
        "relevant_history": context.relevant_history or "Not provided",
        "clinical_notes": context.clinical_notes or "Not provided",
    }

    return {
        "supported_pathologies": supported,
        "clinical_features": clinical_features,
        "concerns": concerns,
        "missing_fields": missing_fields,
    }


class ClinicalFeatureExtractor:
    """Extracts structured and normalized features from ClinicalContextInput."""

    def extract_structured_features(self, context: ClinicalContextInput) -> Dict[str, Any]:
        """Extract structured representation from clinical context."""
        result = extract_clinical_evidence(context)
        return result["clinical_features"]

