"""Localization and anatomical region mapping interface."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class AttentionRegionMapper:
    """
    Translates raw attention heatmap peaks into anatomical lung zones for clinician review:
    - Right / Left hemithorax
    - Upper / Mid / Lower lung zones
    - Perihilar / Retrocardiac / Costophrenic angle regions
    """

    def __init__(self):
        # Anatomical zone definitions normalized to [0.0, 1.0] coordinates
        self.zones = {
            "right_upper": {"x": (0.1, 0.45), "y": (0.1, 0.35), "name": "Right upper lung field"},
            "right_mid": {"x": (0.1, 0.45), "y": (0.35, 0.65), "name": "Right mid lung zone"},
            "right_lower": {"x": (0.1, 0.45), "y": (0.65, 0.90), "name": "Right lower lung base"},
            "left_upper": {"x": (0.55, 0.9), "y": (0.1, 0.35), "name": "Left upper lung field"},
            "left_mid": {"x": (0.55, 0.9), "y": (0.35, 0.65), "name": "Left mid lung zone"},
            "left_lower": {"x": (0.55, 0.9), "y": (0.65, 0.90), "name": "Left lower lung base"},
            "cardiac": {"x": (0.4, 0.65), "y": (0.50, 0.85), "name": "Cardiac silhouette / mediastinum"},
        }

    def identify_dominant_region(self, attention_map: np.ndarray, threshold: float = 0.5) -> Optional[str]:
        """
        Locate center of mass / maximum activation in attention map and map to anatomical zone.
        """
        if attention_map is None or np.all(attention_map == 0):
            return None

        # Coordinates of maximum activation
        max_idx = np.unravel_index(np.argmax(attention_map), attention_map.shape)
        norm_y = max_idx[0] / attention_map.shape[0]
        norm_x = max_idx[1] / attention_map.shape[1]

        for zone_id, zone_info in self.zones.items():
            x_min, x_max = zone_info["x"]
            y_min, y_max = zone_info["y"]
            if x_min <= norm_x <= x_max and y_min <= norm_y <= y_max:
                return zone_info["name"]

        return "Diffuse or non-localized attention pattern"
