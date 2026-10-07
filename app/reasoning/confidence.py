"""Confidence calibration and representation utilities."""

from typing import Dict, Literal, Tuple
import numpy as np


def apply_temperature_scaling(
    prob_dict: Dict[str, float],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Apply temperature scaling to a dict of sigmoid probabilities.

    Converts probabilities → logits, divides by temperature, converts back.
    temperature > 1 → softer (more uncertain) probabilities.
    temperature < 1 → sharper (more confident) probabilities.
    temperature = 1 → no change.

    Args:
        prob_dict: {class_name: probability (0.0-1.0)}
        temperature: Scaling temperature (must be > 0).

    Returns:
        Calibrated probability dict.
    """
    temperature = max(temperature, 1e-4)
    calibrated = {}
    for cls, prob in prob_dict.items():
        # Clamp to avoid log(0)
        prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
        logit = np.log(prob / (1.0 - prob))
        scaled_logit = logit / temperature
        calibrated[cls] = float(1.0 / (1.0 + np.exp(-scaled_logit)))
    return calibrated


def categorize_confidence(probability: float) -> Literal["High", "Moderate", "Low"]:
    """
    Categorize numerical probability score into clinician-oriented certainty tiers.
    """
    if probability >= 0.80:
        return "High"
    elif probability >= 0.50:
        return "Moderate"
    return "Low"


class ConfidenceCalibrator:
    """
    Handles probability calibration and transparent certainty formatting.
    
    SAFETY RULE (AGENTS.md Rule 14):
    Raw probabilities must never be described as medical certainty.
    Always format as 'AI model confidence: [Tier] ([Percentage]%)'.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(temperature, 1e-4)

    def calibrate_logit(self, logit: float) -> float:
        """Apply temperature scaling to raw model logit."""
        scaled_logit = logit / self.temperature
        prob = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(prob)

    def format_confidence_display(self, probability: float) -> str:
        """
        Produce standardized, safe clinician-facing confidence description.
        Example: 'AI model confidence: Moderate (74%)'
        """
        tier = categorize_confidence(probability)
        percent = int(round(probability * 100))
        return f"AI model confidence: {tier} ({percent}%)"
