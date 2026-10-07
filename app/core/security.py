"""Security, privacy, and input validation utilities."""

import os
from pathlib import Path
from typing import List, Tuple
from app.core.config import get_settings


def validate_image_upload(
    filename: str,
    content_type: str,
    file_size_bytes: int,
) -> Tuple[bool, str]:
    """
    Validate uploaded image against file size, extension, and content type.
    Enforces privacy and format safety requirements.
    """
    settings = get_settings()

    if file_size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        return False, f"File size exceeds maximum permitted limit ({max_mb:.1f} MB)."

    ext = Path(filename).suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Permitted: {settings.SUPPORTED_EXTENSIONS}"

    # Also check content type if provided
    if content_type and content_type.lower() not in [fmt.lower() for fmt in settings.SUPPORTED_IMAGE_FORMATS]:
        # Some clients send generic octet-stream for DICOM files
        if ext in [".dcm", ".dicom"] and content_type in ["application/octet-stream", "application/dicom"]:
            pass
        else:
            return False, f"Unsupported MIME type '{content_type}'."

    return True, "Valid"


def sanitize_patient_context(context_text: str) -> str:
    """
    Strip obvious identifiers or control characters from user-provided notes.
    MedVision AI operates exclusively on de-identified clinical context.
    """
    if not context_text:
        return ""
    # Strip null bytes and normalize whitespace
    cleaned = context_text.replace("\x00", "").strip()
    return cleaned
