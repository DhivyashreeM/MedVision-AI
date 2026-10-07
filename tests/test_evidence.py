"""Tests for Evidence Grounding Engine and safety rules.

CRITICAL REQUIREMENT (AGENTS.md Rule 12 & Rule 23):
The evidence engine must have explicit tests preventing unsupported findings
from being presented as established findings.
"""

import pytest
from app.reasoning.evidence import EvidenceEngine


@pytest.fixture
def evidence_engine():
    return EvidenceEngine()


def test_no_evidence_no_asserted_finding(evidence_engine):
    """
    CRITICAL TEST: When neither image evidence nor clinical evidence exists,
    the finding MUST be marked 'insufficient' and MUST NOT be assertable.
    """
    finding = evidence_engine.validate_finding(
        finding_name="Pneumothorax",
        probability=0.75,
        image_evidence=False,
        clinical_evidence=False,
    )

    assert finding.evidence_status == "insufficient"
    assert "Insufficient evidence" in finding.supporting_rationale

    # Must be filtered out of assertable findings
    assertable = evidence_engine.filter_assertable_findings([finding])
    assert len(assertable) == 0


def test_concordant_evidence_supported(evidence_engine):
    """When both image and clinical evidence support the finding, status is 'supported'."""
    finding = evidence_engine.validate_finding(
        finding_name="Pulmonary Opacity",
        probability=0.88,
        image_evidence=True,
        clinical_evidence=True,
        attention_region="Right lower lung base",
        clinical_note_support="fever and productive cough",
    )

    assert finding.evidence_status == "supported"
    assert finding.confidence_level == "High"
    assert finding.model_attention_region == "Right lower lung base"

    assertable = evidence_engine.filter_assertable_findings([finding])
    assert len(assertable) == 1
    assert assertable[0].finding == "Pulmonary Opacity"


def test_image_evidence_without_clinical_context(evidence_engine):
    """Image evidence without explicit clinical corroboration should be presented cautiously."""
    finding = evidence_engine.validate_finding(
        finding_name="Nodule",
        probability=0.62,
        image_evidence=True,
        clinical_evidence=False,
        attention_region="Left upper lung field",
    )

    assert finding.evidence_status == "supported"
    assert finding.confidence_level == "Moderate"
    assert "Clinician review recommended" in finding.supporting_rationale


def test_conflicting_evidence_surfaced(evidence_engine):
    """
    When clinical history suggests finding but image attention shows NO evidence,
    status must be marked 'conflicting', not hallucinated as positive image finding.
    """
    finding = evidence_engine.validate_finding(
        finding_name="Cardiomegaly",
        probability=0.30,
        image_evidence=False,
        clinical_evidence=True,
    )

    assert finding.evidence_status == "conflicting"
    assert "does not demonstrate corroborating radiological abnormality" in finding.supporting_rationale
