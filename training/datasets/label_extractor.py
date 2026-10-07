"""Clinical NLP label extraction module for MedVision AI.

Extracts ground-truth multi-label pathology targets from radiology reports
using medical negation detection and clinical regex patterns.
Derived from chest radiology NLP standards (CheXpert / NegBio principles).
"""

import re
from typing import Dict, List, Optional, Tuple, Any

# Target pathologies for MedVision AI decision support
TARGET_PATHOLOGIES = [
    "Pulmonary_Opacity",
    "Pleural_Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Normal",
]

# Patterns for each pathology
PATHOLOGY_PATTERNS = {
    "Pulmonary_Opacity": [
        r"\b(?:pulmonary\s+)?opacity\b",
        r"\b(?:pulmonary\s+)?opacities\b",
        r"\bpneumonia\b",
        r"\bconsolidation\b",
        r"\binfiltrate\b",
        r"\binfiltrates\b",
        r"\binfiltration\b",
        r"\binfectious\s+process\b",
        r"\bairspace\s+disease\b",
    ],
    "Pleural_Effusion": [
        r"\bpleural\s+effusion\b",
        r"\bpleural\s+effusions\b",
        r"\beffusion\b",
        r"\beffusions\b",
        r"\bblunting\s+of\s+(?:the\s+)?costophrenic\s+angle\b",
    ],
    "Cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\bcardiomegaly\b",
        r"\benlarged\s+cardiac\s+silhouette\b",
        r"\benlarged\s+heart\b",
        r"\bheart\s+(?:size\s+)?is\s+enlarged\b",
        r"\bcardiac\s+enlargement\b",
        r"\bcardiomegaly\s+is\s+seen\b",
    ],
    "Atelectasis": [
        r"\batelectasis\b",
        r"\batelectatic\b",
        r"\bvolume\s+loss\b",
        r"\bdiscoidal\s+atelectasis\b",
        r"\bsubsegmental\s+atelectasis\b",
        r"\bplatelike\s+atelectasis\b",
    ],
    "Normal": [
        r"\bno\s+acute\s+cardiopulmonary\s+process\b",
        r"\bno\s+acute\s+cardiopulmonary\s+abnormality\b",
        r"\bno\s+acute\s+intrathoracic\s+abnormality\b",
        r"\bno\s+acute\s+findings?\b",
        r"\blungs\s+are\s+clear\b",
        r"\bclear\s+lungs\b",
        r"\bunremarkable\s+chest\b",
    ],
}

# Negation triggers preceding the finding
PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bfree\s+of\b",
    r"\bclear\s+of\b",
    r"\bnegative\s+for\b",
    r"\bnot\s+seen\b",
    r"\bdenies\b",
    r"\babsent\b",
    r"\brules?\s+out\b",
    r"\bno\s+evidence\s+of\b",
    r"\bno\s+definite\b",
    r"\bno\s+focal\b",
    r"\bno\s+significant\b",
    r"\bno\s+new\b",
    r"\bno\s+large\b",
]

# Negation triggers following the finding
POST_NEGATION_TRIGGERS = [
    r"\bis\s+absent\b",
    r"\bare\s+absent\b",
    r"\bis\s+ruled\s+out\b",
    r"\bhas\s+resolved\b",
    r"\bis\s+unlikely\b",
    r"\bis\s+negative\b",
]


class ClinicalLabelExtractor:
    """
    Negation-aware clinical label extractor for free-text radiology reports.
    """

    def __init__(self, target_classes: Optional[List[str]] = None):
        self.target_classes = target_classes or TARGET_PATHOLOGIES
        self._compiled_patterns = {
            target: [re.compile(p, re.IGNORECASE) for p in PATHOLOGY_PATTERNS[target]]
            for target in self.target_classes
            if target in PATHOLOGY_PATTERNS
        }
        self._pre_neg = [re.compile(p, re.IGNORECASE) for p in PRE_NEGATION_TRIGGERS]
        self._post_neg = [re.compile(p, re.IGNORECASE) for p in POST_NEGATION_TRIGGERS]

    def _is_negated(self, text: str, match_start: int, match_end: int) -> bool:
        """Check whether a regex match is preceded or followed by negation."""
        # Preceding window (up to 70 characters before match)
        start_win = max(0, match_start - 70)
        pre_text = text[start_win:match_start]
        # Break window at sentence boundaries if any
        if "." in pre_text:
            pre_text = pre_text.split(".")[-1]
        if ";" in pre_text:
            pre_text = pre_text.split(";")[-1]

        for p in self._pre_neg:
            if p.search(pre_text):
                return True

        # Following window (up to 40 characters after match)
        end_win = min(len(text), match_end + 40)
        post_text = text[match_end:end_win]
        if "." in post_text:
            post_text = post_text.split(".")[0]
        for p in self._post_neg:
            if p.search(post_text):
                return True

        return False

    def extract_labels(self, report_text: str) -> Dict[str, int]:
        """
        Extract binary labels (1 = positive presence, 0 = negative / absent / normal)
        for all configured target pathologies.
        """
        if not report_text or not report_text.strip():
            return {target: 0 for target in self.target_classes}

        text = report_text.strip()
        labels = {target: 0 for target in self.target_classes}

        for target, patterns in self._compiled_patterns.items():
            if target == "Normal":
                continue  # Evaluated separately

            for pat in patterns:
                for match in pat.finditer(text):
                    if not self._is_negated(text, match.start(), match.end()):
                        labels[target] = 1
                        break
                if labels[target] == 1:
                    break

        # Normal evaluation:
        # True if explicit normal statement exists AND no positive abnormalities are flagged
        if "Normal" in self.target_classes:
            has_normal_phrase = False
            for pat in self._compiled_patterns.get("Normal", []):
                if pat.search(text):
                    has_normal_phrase = True
                    break

            has_any_abnormality = any(labels[t] == 1 for t in self.target_classes if t != "Normal")
            if has_normal_phrase and not has_any_abnormality:
                labels["Normal"] = 1
            else:
                labels["Normal"] = 0

        return labels

    def extract_label_vector(self, report_text: str) -> List[float]:
        """Return list of float binary indicators aligned with target_classes."""
        label_dict = self.extract_labels(report_text)
        return [float(label_dict[c]) for c in self.target_classes]
