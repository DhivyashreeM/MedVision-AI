"""Prediction execution module."""

from typing import Dict, Any, Optional
import numpy as np
from app.models.model_loader import ModelLoader
from app.models.vision_model import VisionModel


class Predictor:
    """
    Executes model inference and coordinates with model loader.
    Honoring Phase 1 state: Does not fabricate predictions if no model is loaded.
    """

    def __init__(self, model_loader: Optional[ModelLoader] = None):
        self.loader = model_loader or ModelLoader()

    def predict_image(self, preprocessed_tensor: np.ndarray) -> Dict[str, float]:
        """
        Run forward inference on preprocessed tensor.
        Raises NotImplementedError if model is not yet trained/integrated.
        """
        vision_model: VisionModel = self.loader.get_vision_model()
        if not vision_model.is_trained:
            raise NotImplementedError(
                "Prediction engine is not available in Phase 1. "
                "Trained model checkpoint will be integrated in Phase 4."
            )
        return vision_model.predict_probabilities(preprocessed_tensor)
