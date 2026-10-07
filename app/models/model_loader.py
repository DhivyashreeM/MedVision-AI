"""ModelLoader interface for loading vision, text, and multimodal checkpoints."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.vision_model import VisionModel, VisionModelPlaceholder
from app.models.text_model import ClinicalTextModel, ClinicalTextModelPlaceholder
from app.models.fusion_model import FusionModel, FusionModelPlaceholder

logger = get_logger("medvision.models.loader")


class ModelLoader:
    """
    Centralized loader and registry for MedVision AI models.
    Guarantees honest reporting of model availability and prevent hallucinated checkpoints.
    """

    def __init__(self, checkpoints_dir: Optional[Path] = None):
        settings = get_settings()
        self.checkpoints_dir = checkpoints_dir or settings.MODELS_DIR / "checkpoints"
        self._cached_vision_model: Optional[VisionModel] = None
        self._cached_text_model: Optional[ClinicalTextModel] = None
        self._cached_fusion_model: Optional[FusionModel] = None

    def get_vision_model(self, checkpoint_name: Optional[str] = None) -> VisionModel:
        """
        Retrieve loaded vision model or return placeholder if no checkpoint exists.
        """
        if self._cached_vision_model is not None:
            return self._cached_vision_model

        target_file = (
            self.checkpoints_dir / checkpoint_name
            if checkpoint_name
            else self.checkpoints_dir / "baseline_densenet121.pth"
        )

        from app.models.vision_model import ChestXrayVisionModel

        if not target_file.exists():
            logger.info("No vision checkpoint found at %s. Utilizing pretrained DenseNet-121 backbone.", target_file)
            self._cached_vision_model = ChestXrayVisionModel(checkpoint_path=None)
            return self._cached_vision_model

        logger.info("Loading vision checkpoint from %s", target_file)
        self._cached_vision_model = ChestXrayVisionModel(checkpoint_path=str(target_file))
        return self._cached_vision_model

    def get_clinical_text_model(self) -> ClinicalTextModel:
        """Retrieve clinical context model placeholder."""
        if self._cached_text_model is None:
            self._cached_text_model = ClinicalTextModelPlaceholder()
        return self._cached_text_model

    def get_fusion_model(self) -> FusionModel:
        """Retrieve multimodal fusion model placeholder."""
        if self._cached_fusion_model is None:
            self._cached_fusion_model = FusionModelPlaceholder()
        return self._cached_fusion_model

    def get_model_status_summary(self) -> Dict[str, Any]:
        """
        Report availability and training status of each model component honestly.
        """
        vision = self.get_vision_model()
        text = self.get_clinical_text_model()
        fusion = self.get_fusion_model()

        return {
            "vision_model_loaded": vision.is_trained,
            "clinical_text_model_loaded": text.is_trained,
            "fusion_model_loaded": fusion.is_trained,
            "phase": "DenseNet-121 Vision Model & Evidence Pipeline",
            "status": "Ready for inference" if vision.is_trained else "Pretrained ImageNet backbone loaded",
        }

