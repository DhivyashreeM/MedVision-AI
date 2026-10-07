"""Evidence-grounded explanation generation."""

from typing import List, Optional
from app.api.schemas import FindingEvidence, ClinicalContextInput


class ExplanationGenerator:
    """
    Synthesizes verified evidence into clinician-oriented decision support summaries.
    
    ANTI-HALLUCINATION RULE (AGENTS.md Rule 13 & 28):
    Explains ONLY verified findings that have passed the EvidenceEngine check.
    Never invents unverified imaging abnormalities or clinical history.
    """

    def generate_clinical_summary(
        self,
        verified_findings: List[FindingEvidence],
        clinical_context: Optional[ClinicalContextInput] = None,
    ) -> str:
        """
        Produce a concise, structured evidence summary for reviewing physician.
        """
        if not verified_findings:
            return (
                "No focal radiographic abnormalities identified with sufficient AI confidence. "
                "The available image and clinical context do not provide sufficient evidence "
                "for a reliable AI-assisted finding."
            )

        supported_findings = [f for f in verified_findings if f.evidence_status == "supported"]
        conflicting_findings = [f for f in verified_findings if f.evidence_status == "conflicting"]

        lines = []

        if supported_findings:
            lines.append("Supported AI-assisted findings for clinician review:")
            for f in supported_findings:
                region_str = f" in {f.model_attention_region}" if f.model_attention_region else ""
                lines.append(
                    f" - Possible {f.finding.lower()} ({f.confidence_level} confidence: {int(f.confidence * 100)}%){region_str}. "
                    f"Rationale: {f.supporting_rationale}"
                )

        if conflicting_findings:
            lines.append("\nNoted evidence discordance:")
            for f in conflicting_findings:
                lines.append(f" - {f.finding}: {f.supporting_rationale}")

        lines.append(
            "\nRecommendation: Clinical correlation and correlation with prior imaging is recommended."
        )

        return "\n".join(lines)
