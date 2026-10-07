"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application & Environment
    PROJECT_NAME: str = Field(default="MedVision AI", description="Project name")
    VERSION: str = Field(default="0.1.0", description="API version")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    DEBUG: bool = Field(default=True, description="Debug mode")
    API_V1_STR: str = Field(default="/api/v1", description="API v1 prefix")

    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="API server host")
    PORT: int = Field(default=8000, description="API server port")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Allowed HTTP hosts")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"],
        description="Allowed CORS origins",
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="standard", description="Logging format: json or standard")

    # Directory Paths
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
        description="Base project directory",
    )
    DATA_DIR: Path = Field(default=Path("./data"), description="Path to data directory")
    MODELS_DIR: Path = Field(default=Path("./models"), description="Path to models directory")
    OUTPUTS_DIR: Path = Field(default=Path("./outputs"), description="Path to outputs directory")
    CHECKPOINT_PATH: Path = Field(
        default=Path("./models/checkpoints/baseline_densenet121.pth"),
        description="Path to primary model checkpoint",
    )

    # Upload & Image Restrictions
    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=10 * 1024 * 1024,  # 10 MB limit
        description="Maximum image upload size in bytes",
    )
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        default=[
            "image/jpeg",
            "image/png",
            "image/dicom",
            "image/tiff",
            "image/webp",
            "application/dicom",
        ],
        description="MIME types and format identifiers permitted for upload",
    )
    SUPPORTED_EXTENSIONS: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".dcm", ".dicom", ".tif", ".tiff", ".webp"],
        description="Allowed file extensions",
    )

    # Hardware & Model Defaults (Phase 4+ targets)
    DEVICE: str = Field(default="cpu", description="Inference device: cpu or cuda")
    DEFAULT_BATCH_SIZE: int = Field(default=16, description="Batch size for inference/eval")
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.50, description="Decision support threshold for findings"
    )
    CALIBRATION_TEMPERATURE: float = Field(
        default=1.0, description="Temperature scaling factor for confidence calibration"
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    @property
    def is_development(self) -> bool:
        """Check if environment is development."""
        return self.ENVIRONMENT.lower() in ("development", "dev", "local")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
