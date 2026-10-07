"""Tests for vision model architectures, forward pass, and Grad-CAM explainability."""

import io
import numpy as np
import pytest
import torch
from PIL import Image

from app.models.vision_model import DenseNet121Classifier, ChestXrayVisionModel, TARGET_PATHOLOGIES
from app.explainability.gradcam import GradCAM
from app.multimodal.clinical_features import extract_clinical_evidence, ClinicalFeatureExtractor
from app.api.schemas import ClinicalContextInput


def test_densenet121_classifier_forward():
    """Verify DenseNet121Classifier produces correct output logits shape."""
    model = DenseNet121Classifier(num_classes=5, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 5)
    assert not torch.isnan(logits).any()


def test_chest_xray_vision_model_predict():
    """Verify ChestXrayVisionModel outputs valid probabilities for target pathologies."""
    model = ChestXrayVisionModel(checkpoint_path=None, device="cpu")
    dummy_input = torch.randn(3, 224, 224)

    probs = model.predict(dummy_input)
    assert isinstance(probs, dict)
    assert len(probs) == len(TARGET_PATHOLOGIES)

    for pathology in TARGET_PATHOLOGIES:
        assert pathology in probs
        assert 0.0 <= probs[pathology] <= 1.0


def test_gradcam_generation():
    """Verify GradCAM generates non-empty PNG bytes and region description."""
    model = ChestXrayVisionModel(checkpoint_path=None, device="cpu")
    gradcam = GradCAM(model)

    dummy_input = torch.randn(3, 224, 224)
    png_bytes, region_desc = gradcam.generate(
        image_tensor=dummy_input,
        class_idx=0,
        original_size=(256, 256),
    )

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # Verify PNG header
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert isinstance(region_desc, str)
    assert "attention region" in region_desc.lower()


def test_multimodal_clinical_feature_missingness():
    """Verify missing clinical modalities are never converted to false negatives."""
    # Context with no smoking history or age
    context = ClinicalContextInput(symptoms=["cough", "fever"])
    extracted = extract_clinical_evidence(context)

    assert "Pulmonary_Opacity" in extracted["supported_pathologies"]
    assert "age" in extracted["missing_fields"]
    assert "smoking_history" in extracted["missing_fields"]
    assert extracted["clinical_features"]["smoking_history"] == "Not provided"
