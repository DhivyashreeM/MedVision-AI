"""Image preprocessing for MedVision AI inference pipeline."""

import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Decode raw image bytes into a PIL RGB Image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to decode image bytes: {e}") from e


def preprocess_chest_xray(image: Image.Image) -> torch.Tensor:
    """
    Apply standard preprocessing for DenseNet121 inference.
    Returns a (1, 3, 224, 224) float32 tensor ready for model forward pass.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension
