"""Multimodal processing, clinical feature engineering, and fusion."""

from app.multimodal.clinical_features import ClinicalFeatureExtractor
from app.multimodal.text_processing import clean_clinical_notes
from app.multimodal.fusion import MultimodalFusionLayer

__all__ = [
    "ClinicalFeatureExtractor",
    "clean_clinical_notes",
    "MultimodalFusionLayer",
]
