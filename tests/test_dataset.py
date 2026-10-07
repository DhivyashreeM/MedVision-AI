"""Tests for dataset adapter, label extraction NLP, and transforms."""

import numpy as np
from PIL import Image
import pytest
import torch

from training.datasets.label_extractor import ClinicalLabelExtractor
from training.datasets.transforms import get_eval_transform, get_train_transform
from app.inference.preprocessing import preprocess_chest_xray, load_image_from_bytes


def test_label_extractor_positive_findings():
    """Verify NLP extractor detects positive pathologies from report text."""
    extractor = ClinicalLabelExtractor()
    report = "Bilateral pleural effusion with cardiomegaly. Persistent pulmonary opacity in right base."
    labels = extractor.extract_labels(report)

    assert labels["Pleural_Effusion"] == 1
    assert labels["Cardiomegaly"] == 1
    assert labels["Pulmonary_Opacity"] == 1
    assert labels["Normal"] == 0


def test_label_extractor_negation_awareness():
    """Verify negated findings are NOT flagged as positive."""
    extractor = ClinicalLabelExtractor()
    report = "Clear lungs bilaterally. No evidence of pleural effusion, pulmonary opacity, or consolidation."
    labels = extractor.extract_labels(report)

    assert labels["Pleural_Effusion"] == 0
    assert labels["Pulmonary_Opacity"] == 0


def test_label_extractor_normal_detection():
    """Verify clear normal studies are categorized as Normal."""
    extractor = ClinicalLabelExtractor()
    report = "Lungs are clear. Heart size is normal. No acute cardiopulmonary abnormality."
    labels = extractor.extract_labels(report)

    assert labels["Normal"] == 1
    assert labels["Cardiomegaly"] == 0
    assert labels["Pleural_Effusion"] == 0


def test_eval_transforms():
    """Verify eval transforms normalize image to (3, 224, 224) float tensor."""
    transform = get_eval_transform(image_size=224)
    pil_img = Image.new("RGB", (512, 512), color=(128, 128, 128))

    tensor = transform(pil_img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_chest_xray_pipeline():
    """Verify preprocessing handles grayscale and returns (1, 3, 224, 224)."""
    gray_img = Image.new("L", (300, 400), color=100)
    tensor = preprocess_chest_xray(gray_img)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
