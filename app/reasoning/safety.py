"""Clinical safety language validation and mandatory disclaimers."""

import re
from typing import List, Optional, Tuple

MANDATORY_SAFETY_DISCLAIMER = (
    "AI-assisted decision support only. This system does not replace professional medical judgment."
)

FORBIDDEN_ASSERTIONS = [
    r"\bpatient has\b",
    r"\bdefinitely has\b",
    r"\bconfirmed diagnosis\b",
    r"\bguaranteed\b",
    r"\bai diagnosed\b",
    r"\bwe diagnosed\b",
    r"\b100% certain\b",
]


class SafetyValidator:
    """
    Validates that model outputs, summaries, and explanations strictly adhere
    to clinical decision-support language standards (AGENTS.md Rule 16).
    """

    def check_language_safety(self, text: str) -> Tuple[bool, List[str]]:
        """
        Verify text does not contain definitive diagnostic assertions or unsupported claims.
        """
        violations = []
        lower_text = text.lower()
        for pattern in FORBIDDEN_ASSERTIONS:
            if re.search(pattern, lower_text):
                violations.append(f"Forbidden definitive assertion detected: matching '{pattern}'")

        is_safe = len(violations) == 0
        return is_safe, violations

    def enforce_hedged_wording(self, finding_name: str, confidence_tier: str) -> str:
        """
        Format clinician recommendation using hedged decision-support language.
        """
        return f"Possible {finding_name.lower()} detected with {confidence_tier.lower()} model confidence. Doctor, consider reviewing the highlighted region."


def format_safety_summary(
    findings: list,
    quality,
    clinical_concerns: List[str],
    missing_fields: List[str],
    model_is_trained: bool,
) -> Tuple[str, List[str]]:
    """
    Generate a clinician-oriented summary string and recommendations list.

    Args:
        findings: List of FindingEvidence objects (assertable only).
        quality: QualityAssessment object.
        clinical_concerns: Concern strings from clinical feature extraction.
        missing_fields: Fields not provided in clinical context.
        model_is_trained: Whether model was finetuned on chest X-ray data.

    Returns:
        (summary_text, recommendations_list)
    """
    lines = []
    recommendations = []

    # Model status banner
    if not model_is_trained:
        lines.append(
            "⚠️  Note: The vision model is running on pretrained ImageNet weights only "
            "(chest X-ray finetuning not yet applied). Findings require careful clinical review."
        )
        recommendations.append(
            "Model has not been finetuned on chest X-ray data. "
            "Train the model using `python training/train_vision.py` before relying on results."
        )

    # Quality banner
    if not quality.is_acceptable:
        lines.append(
            f"⚠️  Image quality concerns identified: {'; '.join(quality.issues)}. "
            "AI analysis may be less reliable — review original study."
        )
        recommendations.append("Consider re-acquiring the image or reviewing the original study for diagnostic quality.")

    # Findings summary
    if not findings:
        lines.append(
            "No findings met the evidence threshold for AI-assisted reporting. "
            "This may indicate a normal study, or that the model lacks sufficient confidence."
        )
        recommendations.append("If clinical suspicion remains, radiologist review of the original study is recommended.")
    else:
        finding_names = [f.finding for f in findings]
        lines.append(
            f"AI-assisted analysis identified {len(findings)} candidate finding(s) for clinical review: "
            + ", ".join(finding_names) + "."
        )
        for f in findings:
            if f.evidence_status == "conflicting":
                recommendations.append(
                    f"Conflicting evidence for '{f.finding}': image attention and clinical context disagree. "
                    "Radiologist review recommended."
                )
            else:
                recommendations.append(
                    f"Review AI-highlighted region for '{f.finding}' "
                    f"(AI model confidence: {f.confidence_level}, {int(f.confidence * 100)}%)."
                )

    # Clinical concerns
    if clinical_concerns:
        lines.append("Clinical context signals: " + "; ".join(clinical_concerns))

    # Missing fields notice (never invent them)
    if missing_fields:
        lines.append(
            f"Clinical context fields not provided: {', '.join(missing_fields)}. "
            "These were not assumed to be negative."
        )

    # Mandatory disclaimer
    lines.append(MANDATORY_SAFETY_DISCLAIMER)

    summary = " | ".join(lines)
    return summary, recommendations

