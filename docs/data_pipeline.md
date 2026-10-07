# MedVision AI — Data Pipeline & Governance Specification

## 1. Status Notice

> **Current Status:** The model and dataset have not yet been integrated. This document specifies data ingestion policies, leakage prevention, and preprocessing protocols for Phase 2 integration.

---

## 2. Directory Governance

Medical data is partitioned into strictly controlled directories:

```text
data/
├── raw/        # Read-only source datasets (NIH ChestX-ray14 / CheXpert / MIMIC-CXR)
├── processed/  # Preprocessed, normalized tensors and derived cache
├── metadata/   # Patient-split indices, label mappings, and demographic records
└── sample/     # De-identified demonstration and test fixture studies
```

### Critical Storage Rules (AGENTS.md Rule 5)
1. **Source Control Restriction**: Medical data must **NEVER** be committed to Git, GitHub, Docker images, frontend bundles, or public static folders.
2. **Read-Only Raw Data**: Original dataset files are immutable. All processing must output to `data/processed/` or `data/metadata/`.
3. **De-identification**: Only de-identified data without Protected Health Information (PHI) is accepted.

---

## 3. Data Leakage Prevention (AGENTS.md Rule 6)

### Patient-Level Splitting Protocol
Medical datasets commonly contain multiple studies or serial radiographs for the same patient over time. Random image-level splitting leads to severe information leakage between training and validation/test splits.

* **Mandate**: Splitting must be executed on unique `patient_id` values.
* **Ratio**: 70% Train / 15% Validation / 15% Test.
* **Integrity Guarantee**: All images associated with a patient identifier reside exclusively within one partition:
  $$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) = \emptyset$$
  $$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Test}) = \emptyset$$
  $$\text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$$
* **Forbidden Uses of Test Data**: Test subsets must never be consulted during training, hyperparameter search, threshold selection, or calibration.

---

## 4. Preprocessing Specification

1. **Resolution Standardization**: Resized to $224 \times 224$ pixels using bilinear interpolation.
2. **Channel Standardization**: Conversion to 3-channel RGB representation.
3. **Intensity Normalization**: Scaled to $[0.0, 1.0]$ and standardized via ImageNet population statistics:
   * $\text{Mean} = [0.485, 0.456, 0.406]$
   * $\text{Std} = [0.229, 0.224, 0.225]$

---

## 5. Clinical Context Handling (AGENTS.md Rule 10)

### Missing Information Policy
Missing patient history or clinical parameters must **never** be converted into negative indicators:
* If smoking status is not provided, record `"Not provided"` (do **not** assume non-smoker).
* If temperature or laboratory values are absent, record `"Not measured"`.
* Feature vectorizers utilize explicit presence/absence masks to avoid imputation bias.
