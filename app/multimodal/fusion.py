"""Multimodal feature fusion layer interface."""

from typing import Tuple
import numpy as np


class MultimodalFusionLayer:
    """
    Combines visual embeddings and clinical context representations.
    Supports early concatenation, gated fusion, or late decision blending.
    
    CRITICAL RULE (AGENTS.md Rule 11):
    Multimodal improvement must be experimentally verified in Phase 9
    against image-only and clinical-only baselines.
    """

    def __init__(self, visual_dim: int = 1024, clinical_dim: int = 16):
        self.visual_dim = visual_dim
        self.clinical_dim = clinical_dim
        self.total_dim = visual_dim + clinical_dim

    def concatenate_features(
        self,
        visual_vec: np.ndarray,
        clinical_vec: np.ndarray,
    ) -> np.ndarray:
        """Concatenate flattened visual embedding with clinical feature vector."""
        v_flat = visual_vec.flatten()
        c_flat = clinical_vec.flatten()
        return np.concatenate([v_flat, c_flat])
