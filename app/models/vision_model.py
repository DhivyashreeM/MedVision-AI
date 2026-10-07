"""DenseNet121 Vision Model for Chest X-ray Abnormality Classification.

Architecture: DenseNet121 (pretrained on ImageNet) with custom classification head.
Target layer for Grad-CAM: features.denseblock4.denselayer16.conv2

Reference: Rajpurkar et al., CheXNet (2017) — DenseNet121 chest X-ray classification.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from app.core.logging import get_logger

logger = get_logger("medvision.models.vision")

# Pathologies aligned with dataset label extractor
TARGET_PATHOLOGIES = [
    "Pulmonary_Opacity",
    "Pleural_Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Normal",
]


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 with custom multi-label classification head.
    Uses pretrained ImageNet weights, replaces the final FC layer.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, num_classes)
        self.features = backbone.features
        self.classifier = backbone.classifier
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits


class VisionModel:
    """Base interface for vision models in MedVision AI."""

    def __init__(self, *args, **kwargs):
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class VisionModelPlaceholder(VisionModel):
    """Placeholder vision model used prior to checkpoint integration."""

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Vision model checkpoint not yet loaded.")


class ChestXrayVisionModel(VisionModel):
    """
    High-level wrapper around DenseNet121Classifier.
    Manages loading, inference, and Grad-CAM target layer access.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.target_classes = TARGET_PATHOLOGIES
        self.num_classes = len(self.target_classes)
        self._is_trained = False
        self._checkpoint_path = checkpoint_path

        # Build model
        self.model = DenseNet121Classifier(
            num_classes=self.num_classes,
            pretrained=True,  # Always start with ImageNet weights
        )
        self.model.to(self.device)
        self.model.eval()

        # Attempt to load finetuned checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        """Load finetuned weights from checkpoint file."""
        import os
        if not os.path.exists(path):
            logger.warning("Checkpoint not found at %s — using ImageNet pretrained backbone only.", path)
            return
        try:
            state = torch.load(path, map_location=self.device)
            # Support checkpoints saved as {'model_state_dict': ..., ...}
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state, strict=True)
            self._is_trained = True
            logger.info("Finetuned checkpoint loaded from %s", path)
        except Exception as e:
            logger.warning("Failed to load checkpoint from %s: %s — using pretrained backbone.", path, str(e))

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict(self, image_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Run forward pass and return sigmoid probabilities per class.

        Args:
            image_tensor: (C, H, W) or (1, C, H, W) float tensor or numpy array.

        Returns:
            Dict mapping pathology name → probability (0.0-1.0).
        """
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor).float()

        self.model.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        return {cls: float(probs[i]) for i, cls in enumerate(self.target_classes)}

    predict_probabilities = predict


    def get_feature_layer(self) -> nn.Module:
        """Return the DenseNet final dense block for Grad-CAM hooks."""
        return self.model.features.denseblock4

    def get_feature_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract feature map tensor from final dense block (used by Grad-CAM)."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        feature_map = None

        def hook_fn(module, input, output):
            nonlocal feature_map
            feature_map = output

        handle = self.model.features.denseblock4.register_forward_hook(hook_fn)
        with torch.no_grad():
            self.model(image_tensor)
        handle.remove()

        return feature_map
