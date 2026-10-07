"""De-identified clinical text processing utilities."""

import re
from typing import List


def clean_clinical_notes(notes: str) -> str:
    """
    Normalize clinical notes, stripping extra whitespace and control codes.
    Maintains medical terminology integrity.
    """
    if not notes:
        return ""
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", notes)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords_from_notes(notes: str) -> List[str]:
    """
    Extract documented symptoms or findings mentioned in clinical text.
    """
    text = clean_clinical_notes(notes).lower()
    clinical_keywords = [
        "fever",
        "cough",
        "dyspnea",
        "shortness of breath",
        "hypoxia",
        "chest pain",
        "crackles",
        "wheeze",
        "rales",
        "consolidation",
        "infiltrate",
        "effusion",
    ]
    found = [kw for kw in clinical_keywords if kw in text]
    return found
