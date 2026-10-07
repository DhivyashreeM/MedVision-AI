"""Image conversion and hashing utilities."""

import hashlib
import io
from typing import Union
import numpy as np
from PIL import Image


def array_to_png_bytes(image: Union[np.ndarray, Image.Image]) -> bytes:
    """Encode an image or numpy array into PNG format bytes."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image bytes for study deduplication and tracking."""
    return hashlib.sha256(image_bytes).hexdigest()
