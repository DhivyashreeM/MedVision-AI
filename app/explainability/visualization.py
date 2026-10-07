"""Visualization utilities for attention heatmaps and clinician overlays."""

from typing import Union
import numpy as np
from PIL import Image
import matplotlib.cm as cm


def overlay_attention_heatmap(
    base_image: Union[Image.Image, np.ndarray],
    attention_map: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Superimpose attention heatmap onto chest X-ray image for clinician review.
    
    Args:
        base_image: Original chest X-ray (PIL Image or uint8 array).
        attention_map: 2D float array [0, 1] of attention values.
        alpha: Blending weight for heatmap overlay (0.0 to 1.0).
        colormap_name: Matplotlib colormap (default 'jet').
        
    Returns:
        PIL.Image: RGB blended visualization.
    """
    if isinstance(base_image, np.ndarray):
        pil_img = Image.fromarray(base_image.astype(np.uint8)).convert("RGB")
    else:
        pil_img = base_image.convert("RGB")

    width, height = pil_img.size

    # Resize attention map to image dimensions
    heatmap_pil = Image.fromarray((attention_map * 255).astype(np.uint8)).resize(
        (width, height), resample=Image.Resampling.BILINEAR
    )
    norm_heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply colormap
    cmap = cm.get_cmap(colormap_name)
    colored_heatmap = cmap(norm_heatmap)[:, :, :3]  # Drop alpha channel
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    # Blend images
    base_arr = np.array(pil_img, dtype=np.float32)
    blended = (1 - alpha) * base_arr + alpha * colored_heatmap_uint8
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_uint8)
