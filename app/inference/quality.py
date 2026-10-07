"""Image quality assessment for MedVision AI.

Evaluates chest X-ray diagnostic quality criteria before inference.
AGENTS.md Rule 15: If quality is inadequate, warn the user.
"""

import io
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from app.api.schemas import QualityAssessment
from app.core.logging import get_logger

logger = get_logger("medvision.inference.quality")

# Quality thresholds
MIN_DIMENSION = 64          # px — reject extremely small images
MIN_BLUR_SCORE = 25.0       # Laplacian variance — below this = blurry
MIN_CONTRAST_RANGE = 30     # Pixel intensity range — below this = very low contrast
MIN_MEAN_BRIGHTNESS = 10    # Mean pixel value — below this = underexposed
MAX_MEAN_BRIGHTNESS = 245   # Mean pixel value — above this = overexposed


def _laplacian_variance(gray_array: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness/blur metric."""
    img = Image.fromarray(gray_array)
    lap = img.filter(ImageFilter.FIND_EDGES)
    lap_arr = np.array(lap, dtype=np.float64)
    return float(np.var(lap_arr))


def assess_image_quality(
    image: Image.Image,
    min_dimension: int = MIN_DIMENSION,
) -> QualityAssessment:
    """
    Perform multi-criteria image quality assessment on a PIL Image.

    Checks:
    - Dimensions: rejects tiny images
    - Blur: Laplacian variance sharpness metric
    - Contrast: pixel intensity range
    - Brightness: mean pixel intensity (under/over exposure)

    Returns QualityAssessment with is_acceptable flag, scores, and issues list.
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Ensure RGB for consistent analysis, then derive grayscale via PIL
    # (avoids large intermediate float32 allocation from np.mean(arr, axis=2))
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    arr = np.array(image, dtype=np.uint8)

    # Use PIL's native grayscale conversion — ITU-R 601 luma weights, no extra alloc
    gray_pil = image.convert("L")
    gray = np.array(gray_pil, dtype=np.float32)

    # 1. Dimension check
    if w < min_dimension or h < min_dimension:
        issues.append(f"Image too small ({w}x{h}px). Minimum {min_dimension}x{min_dimension}px required.")

    # 2. Blur check
    blur_score = _laplacian_variance(gray.astype(np.uint8))
    if blur_score < MIN_BLUR_SCORE:
        issues.append(f"Image appears blurry (sharpness score: {blur_score:.1f}). Minimum: {MIN_BLUR_SCORE}.")

    # 3. Contrast check
    contrast_range = float(gray.max() - gray.min())
    if contrast_range < MIN_CONTRAST_RANGE:
        issues.append(f"Very low contrast detected (range: {contrast_range:.1f}). Image may not be diagnostic.")

    # 4. Brightness check
    mean_brightness = float(gray.mean())
    if mean_brightness < MIN_MEAN_BRIGHTNESS:
        issues.append(f"Image appears underexposed (mean brightness: {mean_brightness:.1f}).")
    elif mean_brightness > MAX_MEAN_BRIGHTNESS:
        issues.append(f"Image appears overexposed (mean brightness: {mean_brightness:.1f}).")

    is_acceptable = len(issues) == 0

    warning_text: Optional[str] = None
    if not is_acceptable:
        warning_text = (
            "Image quality may limit reliable AI analysis. "
            "Issues detected: " + "; ".join(issues) + " "
            "Please review the original study."
        )
    elif blur_score < MIN_BLUR_SCORE * 2:
        warning_text = "Image sharpness is marginal. AI analysis may be less reliable."

    return QualityAssessment(
        is_acceptable=is_acceptable,
        dimensions=[h, w, arr.shape[2] if arr.ndim == 3 else 1],
        blur_score=round(blur_score, 2),
        contrast_score=round(contrast_range, 2),
        brightness_score=round(mean_brightness, 2),
        issues=issues,
        warning=warning_text,
    )
