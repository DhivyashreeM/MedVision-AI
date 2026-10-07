# MedVision AI — Comprehensive System & Performance Audit

**Audit Date:** October 2026  
**Auditor:** Lead ML & Medical AI Systems Engineer  
**Reference Mandates:** AGENTS.md, BRAND_GUIDELINES.md, HNX26PSI05 Guidelines  

---

## 1. Executive Summary & Root Cause Analysis

### User-Reported Issue
> *"the output doesnt change at all is there any problem then, correct it"*

### Critical Root Cause Discovered
During comprehensive system trace analysis of `app/inference/pipeline.py` and `app/reasoning/evidence.py`, the reason the output "does not change at all" across different uploaded chest X-rays was identified:

1. **Uniform Arbitrary Reporting Threshold (`0.35`):**
   In `app/inference/pipeline.py:41`, `REPORTING_THRESHOLD = 0.35` was hardcoded. Because uncalibrated / weakly-trained network logits across all classes produce probabilities in the `0.35 - 0.70` range on almost every input, **all four abnormal pathologies (`Pulmonary_Opacity`, `Pleural_Effusion`, `Cardiomegaly`, `Atelectasis`) exceeded the 0.35 cutoff on every single radiograph**.

2. **Circular Evidence Verification in `pipeline.py`:**
   In `pipeline.py:171`, the pipeline set:
   ```python
   image_ev = prob >= REPORTING_THRESHOLD  # Always True!
   ```
   Because `image_ev` was trivially evaluated as `True` for every pathology exceeding 0.35, `EvidenceEngine.validate_finding()` classified every single pathology as `evidence_status = "supported"`. Consequently, **every radiograph received the exact same four finding cards**, regardless of actual anatomical presentation.

3. **Streamlit State Retention on Preset Switching:**
   In `frontend/streamlit_app.py`, selecting different preset cases from the sidebar failed to refresh user inputs (`st.number_input`, `st.multiselect`) because widget state was keyed statically, causing clinical context from previous runs to persist across preset selections.

4. **Under-Trained Vision Backbone:**
   The initial baseline was trained on only 60 samples for 1 epoch, resulting in Macro AUROC of 0.4845. The network had not learned sufficient discriminative feature representations across diverse CXR patterns.

---

## 2. Component-by-Component Audit Findings

| Component | Current State | Issues Identified | Remediation Action |
| :--- | :--- | :--- | :--- |
| **Data Pipeline** | 21,443 rows in 2 Parquet shards | No patient IDs; potential report duplication across splits | Audit report hashes, compute class distributions, save `dataset_statistics.json` |
| **Label Extraction** | Clinical NLP via regex/keyword mapping | Negations handled, but uncertainty / past history not fully audited | Create `docs/label_quality.md` with explicit validation cases |
| **Vision Model** | DenseNet-121 with ImageNet backbone | Under-trained; macro AUROC 0.4845 | Implement improved training with cosine annealing, stronger augmentation, and class reweighting |
| **Thresholding** | Fixed 0.35 across all classes | Causes false positives on every study; ignores class imbalance | Compute empirical validation-set thresholds; save to `outputs/metrics/thresholds.json` |
| **Calibration** | Fixed T=1.0 scaling | No real calibration on validation logits | Fit empirical temperature scaling on validation set; save to `outputs/metrics/calibration.json` |
| **Evidence Engine** | Bypassed by `image_ev = prob >= 0.35` | Weak low-confidence findings asserted without clinical backing | Enforce strict suppression: findings below optimal threshold without clinical concordance marked `insufficient` |
| **Streamlit Workstation** | Functional but state persistence bug | Preset selection doesn't refresh input form; identical 4 cards shown | Add session state refresh keys; show distinct outputs per case; add Empty, Poor-Quality, and Normal states |
| **FastAPI Backend** | 4 endpoints functioning | Does not serve optimal thresholds or calibration metadata | Expose `/api/v1/capabilities` and `/api/v1/metrics`; update `/analyze` with class-specific thresholds |
| **MCP Integration** | Not yet implemented | Standalone CLI/HTTP only | Implement standardized MedVision MCP tool service for IDE/agent workflows |

---

## 3. Systematic Action Plan

- **Phase B:** Dataset validation, duplicate checking, distribution export (`dataset_statistics.json`).
- **Phase C:** Label extraction auditing and edge-case validation (`docs/label_quality.md`).
- **Phase D & E:** Improved vision training with data augmentation, focal/BCE class weighting, learning rate scheduler.
- **Phase F & G:** Class-specific threshold optimization and validation-set temperature calibration.
- **Phase H & I:** Evidence Engine hardening (true suppression of unsupported findings).
- **Phase J & K:** Streamlit UI state management fix & MCP service integration.
- **Phase L & M:** Comprehensive testing and final documentation.
