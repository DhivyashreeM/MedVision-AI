"""Grad-CAM explainability for MedVision AI.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for
DenseNet121 to produce visual attention heatmaps over chest X-ray images.

IMPORTANT — AGENTS.md Rule 9:
    Grad-CAM represents model attention / importance.
    Do NOT describe a Grad-CAM heatmap as a clinically verified lesion segmentation.
    Use terminology: 'Model attention region', 'Region influencing the prediction',
    'AI-highlighted region for review'.
"""

import io
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.core.logging import get_logger

logger = get_logger("medvision.explainability.gradcam")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for DenseNet121.

    Usage:
        cam = GradCAM(model_wrapper)
        heatmap_png_bytes, region_desc = cam.generate(image_tensor, class_idx=0)
    """

    def __init__(self, model_wrapper):
        """
        Args:
            model_wrapper: ChestXrayVisionModel instance.
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

    def _register_hooks(self, target_layer: nn.Module):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)
        return fwd_handle, bwd_handle

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: int,
        original_size: Tuple[int, int] = (512, 512),
    ) -> Tuple[bytes, str]:
        """
        Generate a Grad-CAM heatmap for a given class index.

        Args:
            image_tensor: (C, H, W) or (1, C, H, W) preprocessed tensor.
            class_idx: Index of the target class to visualize.
            original_size: (height, width) of the original image for upsampling.

        Returns:
            (heatmap_png_bytes, region_description)
            heatmap_png_bytes: PNG image bytes of the overlay heatmap.
            region_description: Text description of the most activated region.
        """
        self.model.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(False)

        target_layer = self.model.features.denseblock4
        fwd_handle, bwd_handle = self._register_hooks(target_layer)

        try:
            # Forward pass with gradient enabled
            with torch.enable_grad():
                inp = image_tensor.clone().requires_grad_(True)
                logits = self.model(inp)
                # Zero existing grads
                self.model.zero_grad()
                # Backward on the target class score
                class_score = logits[0, class_idx]
                class_score.backward()

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

        if self._gradients is None or self._activations is None:
            logger.warning("Grad-CAM hooks did not capture gradients/activations.")
            return b"", "Region analysis unavailable."

        # Global Average Pool gradients over spatial dims → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam_map = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam_map = F.relu(cam_map)

        # Normalize to [0, 1]
        cam_min = cam_map.min()
        cam_max = cam_map.max()
        if cam_max > cam_min:
            cam_map = (cam_map - cam_min) / (cam_max - cam_min)
        else:
            cam_map = torch.zeros_like(cam_map)

        # Upsample to original image size
        cam_resized = F.interpolate(
            cam_map,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()  # (H, W)

        # Generate region description from peak activation location
        region_desc = self._describe_region(cam_resized, original_size)

        # Render heatmap as PNG overlay
        heatmap_bytes = self._render_heatmap(cam_resized)

        return heatmap_bytes, region_desc

    def _describe_region(
        self,
        cam: np.ndarray,
        image_size: Tuple[int, int],
    ) -> str:
        """
        Convert peak activation coordinates to anatomical region description.
        Uses simple quadrant-based anatomical mapping for frontal chest X-rays.
        """
        h, w = image_size
        peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)

        # Vertical thirds: Upper / Middle / Lower
        if peak_y < h / 3:
            vert = "upper"
        elif peak_y < 2 * h / 3:
            vert = "mid"
        else:
            vert = "lower"

        # Horizontal halves: Left / Right (radiological convention: right = patient's right)
        if peak_x < w / 2:
            horiz = "left"
        else:
            horiz = "right"

        return f"{vert.capitalize()} {horiz} lung field (AI model attention region)"

    def _render_heatmap(self, cam: np.ndarray) -> bytes:
        """
        Convert normalized CAM array to a coloured PNG heatmap image.
        Uses a simple red-channel thermal colormap without requiring matplotlib.
        """
        # Scale to uint8
        cam_uint8 = (cam * 255).astype(np.uint8)

        # Build RGBA thermal colormap: low=blue, mid=green, high=red
        h, w = cam_uint8.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 3] = 160  # Semi-transparent

        # Red channel: high activation
        rgba[:, :, 0] = cam_uint8
        # Green channel: mid activation
        rgba[:, :, 1] = np.clip(255 - np.abs(cam_uint8.astype(int) - 128) * 2, 0, 255).astype(np.uint8)
        # Blue channel: low activation
        rgba[:, :, 2] = (255 - cam_uint8).astype(np.uint8)

        overlay_img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        return buf.getvalue()


# Alias for backwards compatibility
GradCAMExplainer = GradCAM

