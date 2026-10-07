"""Tests for configuration loading and validation."""

import pytest
from app.core.config import Settings, get_settings


def test_default_settings_load():
    """Verify default configuration values."""
    settings = get_settings()
    assert settings.PROJECT_NAME == "MedVision AI"
    assert settings.VERSION == "0.1.0"
    assert settings.MAX_UPLOAD_SIZE_BYTES > 0
    assert ".jpg" in settings.SUPPORTED_EXTENSIONS
    assert ".dcm" in settings.SUPPORTED_EXTENSIONS
    assert settings.is_development is True


def test_invalid_log_level_rejected():
    """Verify that an invalid log level raises a validation error."""
    with pytest.raises(ValueError):
        Settings(LOG_LEVEL="INVALID_LEVEL")


def test_settings_caching():
    """Verify get_settings returns identical cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
