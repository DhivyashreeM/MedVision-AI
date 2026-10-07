"""Clinical context and text processing model interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class ClinicalTextModel(ABC):
    """
    Abstract Base Class for encoding structured and unstructured clinical context
    (e.g., patient demographics, vital signs, reported symptoms, clinical history).
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Return whether context model is trained and ready."""
        return self._is_trained

    @abstractmethod
    def encode_context(self, context_data: Dict[str, Any]) -> np.ndarray:
        """
        Encode clinical dictionary into numerical vector representation.
        Returns:
            np.ndarray: Vector representation of shape (embedding_dim,).
        """
        pass

    @abstractmethod
    def extract_symptom_prior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Calculate prior probability indicators based on clinical presentation.
        Returns:
            Dict[str, float]: Finding priors (0.0 to 1.0).
        """
        pass


class ClinicalTextModelPlaceholder(ClinicalTextModel):
    """
    Phase 1 placeholder for clinical context model.
    """

    def encode_context(self, context_data: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError(
            "Clinical context encoder has not yet been integrated. "
            "Context processing is scheduled for Phase 8."
        )

    def extract_symptom_prior(self, symptoms: List[str]) -> Dict[str, float]:
        raise NotImplementedError(
            "Symptom prior extraction is not yet configured. "
            "Evidence rules will be implemented during Phase 8 & 10."
        )
