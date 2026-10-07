"""Tests for image quality assessment module."""

import numpy as np
from PIL import Image
import pytest

from app.inference.quality import assess_image_quality


def test_quality_normal_image():
    """Verify standard high-contrast image passes quality check."""
    # Create synthetic 256x256 image with good contrast and edges
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[64:192, 64:192] = 200
    arr[100:150, 100:150] = 50
    img = Image.fromarray(arr)

    quality = assess_image_quality(img)
    assert quality.is_acceptable is True
    assert len(quality.issues) == 0
    assert quality.dimensions == [256, 256, 3]
    assert quality.contrast_score >= 30


def test_quality_tiny_image_rejected():
    """Verify undersized image is flagged."""
    tiny = Image.new("RGB", (32, 32), color=(128, 128, 128))
    quality = assess_image_quality(tiny, min_dimension=64)
    assert quality.is_acceptable is False
    assert any("too small" in issue.lower() for issue in quality.issues)


def test_quality_low_contrast_image_flagged():
    """Verify completely flat/blank image fails contrast check."""
    flat = Image.new("RGB", (256, 256), color=(120, 120, 120))
    quality = assess_image_quality(flat)
    assert quality.is_acceptable is False
    assert any("contrast" in issue.lower() for issue in quality.issues)


def test_quality_underexposed_image_flagged():
    """Verify very dark image fails brightness check."""
    dark = Image.new("RGB", (256, 256), color=(5, 5, 5))
    quality = assess_image_quality(dark)
    assert quality.is_acceptable is False
    assert any("underexposed" in issue.lower() for issue in quality.issues)
