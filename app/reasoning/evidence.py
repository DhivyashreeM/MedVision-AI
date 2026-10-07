"""Evidence grounding engine for MedVision AI.

CRITICAL SAFETY PRINCIPLE (AGENTS.md Rule 1 & Rule 12):
> Prediction + Localization + Evidence + Confidence
> No evidence → no asserted finding.

Every reported finding must be traceable to:
1. Evidence in the medical image, OR
2. Evidence in the supplied clinical context, OR
3. Both.
"""

from typing import List, Optional, Literal, Dict, Any
from app.api.schemas import FindingEvidence


class EvidenceEngine:
    """
    Validates candidate findings against image and clinical evidence traces.
    Enforces that unsupported findings are NEVER presented as established findings.
    """

    def validate_finding(
        self,
        finding_name: str,
        probability: float,
        image_evidence: bool,
        clinical_evidence: bool,
        attention_region: Optional[str] = None,
        clinical_note_support: Optional[str] = None,
    ) -> FindingEvidence:
        """
        Evaluate and ground a candidate finding against verified evidence.

        `image_evidence` (bool) is set by the pipeline layer:
            - True  → probability cleared the STRONG_EVIDENCE_THRESHOLD (≥0.60)
            - False → probability only cleared the REPORTING_THRESHOLD (≥0.50),
                      i.e. the model sees a weak-to-moderate signal.

        Rules (AGENTS.md §12 — Evidence-Grounding Engine):
        1. No image evidence + no clinical evidence → 'insufficient' (not asserted).
        2. Strong image evidence + clinical evidence → 'supported'.
        3. Strong image evidence, no clinical evidence → 'supported' with caveats.
        4. Weak image evidence + clinical evidence → 'conflicting' (both sources
           present but image signal is too weak to independently support finding).
        5. Weak image evidence, no clinical evidence → 'insufficient' (not asserted).
        """
        # Determine confidence category
        if probability >= 0.80:
            confidence_level: Literal["High", "Moderate", "Low"] = "High"
        elif probability >= 0.60:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        # Determine evidence status using the two-flag system
        if image_evidence and clinical_evidence:
            evidence_status: Literal["supported", "insufficient", "conflicting", "not_detected"] = "supported"
            region_desc = f" localized to {attention_region}" if attention_region else ""
            rationale = (
                f"Finding supported by concordant evidence: Model visual attention{region_desc} "
                f"aligns with clinical presentation"
                f"{f' ({clinical_note_support})' if clinical_note_support else ''}. "
                "Doctor, consider reviewing the highlighted region."
            )

        elif image_evidence and not clinical_evidence:
            evidence_status = "supported"
            region_desc = f" in {attention_region}" if attention_region else ""
            rationale = (
                f"Image evidence observed{region_desc} without explicit corroborating clinical "
                "context. AI model confidence is above the strong-evidence threshold. "
                "Clinician review recommended to determine clinical significance."
            )

        elif not image_evidence and clinical_evidence:
            # Weak imaging signal but clinical context supports the finding
            evidence_status = "conflicting"
            rationale = (
                f"Clinical context suggests potential {finding_name}, but the imaging signal is "
                "below the strong-evidence threshold. The model detected only a weak visual "
                "pattern. Review the original study carefully."
            )

        else:
            # Weak imaging signal + no clinical corroboration → suppress
            evidence_status = "insufficient"
            rationale = (
                f"Insufficient evidence for a reliable AI-assisted finding of {finding_name}. "
                "Image model signal is below strong-evidence threshold and no corroborating "
                "clinical context was provided. Finding is suppressed per safety policy."
            )

        return FindingEvidence(
            finding=finding_name,
            confidence=round(probability, 4),
            confidence_level=confidence_level,
            image_evidence=image_evidence,
            clinical_evidence=clinical_evidence,
            evidence_status=evidence_status,
            model_attention_region=attention_region,
            supporting_rationale=rationale,
        )

    def filter_assertable_findings(
        self, candidate_findings: List[FindingEvidence]
    ) -> List[FindingEvidence]:
        """
        Enforce AGENTS.md Rule 12: Unsupported findings MUST NOT be presented as
        established findings.  Only 'supported' and 'conflicting' statuses are
        surfaced to the clinician; 'insufficient' / 'not_detected' are dropped.
        """
        assertable = [
            f for f in candidate_findings if f.evidence_status in ("supported", "conflicting")
        ]
        return assertable
