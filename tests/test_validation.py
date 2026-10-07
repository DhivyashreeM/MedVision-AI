"""Tests for input validation, quality screening, and safety language enforcement."""

import pytest
import numpy as np
from PIL import Image
from app.core.security import validate_image_upload
from app.inference.quality import assess_image_quality
from app.reasoning.safety import SafetyValidator
from app.multimodal.clinical_features import ClinicalFeatureExtractor
from app.api.schemas import ClinicalContextInput


def test_oversized_upload_rejected():
    """Verify upload validator rejects files exceeding byte limit."""
    huge_size = 20 * 1024 * 1024  # 20 MB (limit is 10 MB)
    valid, msg = validate_image_upload("chest_xray.png", "image/png", huge_size)
    assert valid is False
    assert "exceeds maximum permitted limit" in msg


def test_unsupported_file_extension_rejected():
    """Verify unsupported file extensions are rejected."""
    valid, msg = validate_image_upload("report.pdf", "application/pdf", 1024)
    assert valid is False
    assert "Unsupported file extension" in msg


def test_image_quality_small_dimension_flagged():
    """Verify tiny images below diagnostic threshold are flagged with warning."""
    tiny_img = Image.new("L", (32, 32), color=128)
    quality = assess_image_quality(tiny_img, min_dimension=128)
    assert quality.is_acceptable is False
    assert len(quality.issues) > 0
    assert quality.warning is not None
    assert "Image quality may limit reliable AI analysis" in quality.warning


def test_safety_validator_forbids_definitive_diagnostic_language():
    """Verify SafetyValidator flags forbidden definitive claims."""
    validator = SafetyValidator()

    bad_sentence = "The patient definitely has pneumonia diagnosed by AI."
    is_safe, violations = validator.check_language_safety(bad_sentence)
    assert is_safe is False
    assert len(violations) > 0

    safe_sentence = "Possible pulmonary opacity detected. Doctor, consider reviewing the highlighted region."
    is_safe, violations = validator.check_language_safety(safe_sentence)
    assert is_safe is True
    assert len(violations) == 0


def test_missing_clinical_context_never_assumed_negative():
    """Verify omission of smoking history is reported as 'Not provided', not 'non-smoker'."""
    extractor = ClinicalFeatureExtractor()
    context = ClinicalContextInput(age=55, symptoms=["cough"])
    features = extractor.extract_structured_features(context)

    assert features["smoking_history"] == "Not provided"
    assert features["smoking_history"] != "non-smoker"
