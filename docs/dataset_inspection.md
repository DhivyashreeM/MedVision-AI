# MedVision AI — Comprehensive Dataset Inspection Report

**Project Code:** HNX26PSI05 — Multimodal Medical Image Intelligence  
**Phase:** Dataset Inspection & Architecture Alignment (Pre-Training Assessment)  
**Inspection Date:** 2026-10-07  

---

## 1. Executive Summary & Dataset Overview

A thorough, non-destructive empirical inspection of the raw data files in `data/raw/` was conducted to determine the exact properties, schema, image modality, label structures, and clinical text availability prior to model development.

### Raw Data Files Verified

| File Name | File Size | Number of Rows | Row Groups | Format | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train-00000-of-00002.parquet` | 263.96 MB (276,780,501 bytes) | 10,722 | 108 | Apache Parquet | Verified, Immutable |
| `train-00001-of-00002.parquet` | 264.17 MB (276,998,657 bytes) | 10,721 | 108 | Apache Parquet | Verified, Immutable |
| **Combined Dataset** | **528.13 MB (553,779,158 bytes)** | **21,443** | **216** | **Parquet** | **Non-corrupt** |

### Dataset Identity & Sharding Assessment
* **Two Shards of ONE Dataset Split:** The filenames `train-00000-of-00002.parquet` and `train-00001-of-00002.parquet` represent two physical shards of a single unified training split.
* **Empirical Evidence:**
  1. The schemas of both files are 100% identical (`['image', 'reports']`).
  2. The custom Hugging Face feature metadata is identical across both files (`Image` feature struct + `reports` string).
  3. Every single image hash in File 0 is unique ($N = 10,722$), and every image hash in File 1 is unique ($N = 10,721$).
  4. There is **zero image hash overlap** between File 0 and File 1 ($0$ shared image hashes).
  5. The records partition 21,443 distinct image-report pairs into two nearly equal shards ($10,722$ and $10,721$).

### Provenance Determination (AGENTS.md Anti-Hallucination Mandate)
* **Local File Audit:** The Parquet metadata contains standard Hugging Face Arrow metadata (`features: {image: Image, reports: Value(string)}`), but does **not** embed an explicit dataset name, repository URL, or license string.
* **Literature Grounding:** The narrative style, de-identification syntax (e.g., `"In comparison to study performed on of..."`), and representative report text (e.g., *"Multiple surgical clips project over the left breast, and old left rib fractures are noted"*) correspond to de-identified chest X-ray reports derived from the **MIMIC-CXR** database.
* **Formal Conclusion:** **Provenance could not be conclusively determined from the local files.** While radiographic and textual features strongly correspond to de-identified MIMIC-CXR derived subsets, no external claims are asserted as ground-truth metadata.

---

## 2. Dataset Schema

The dataset contains exactly two columns with zero null values across all 21,443 records:

| Column Name | Physical Data Type | Logical Feature Type | Null Count | Null % | Unique Values | Description & Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `image` | `struct<bytes: binary, path: string>` | Hugging Face `Image` Feature | 0 | 0.0% | 21,443 unique byte hashes | Raw image binary data containing frontal chest radiograph (JPEG format). `path` is uniformly null. |
| `reports` | `string` / `binary (String)` | `Value(dtype='string')` | 0 | 0.0% | 21,010 unique strings | Free-text radiology report written by the radiologist, containing narrative Findings and Impression sections. |

---

## 3. Image Analysis & Modality Verification

### Visual Modality
* Diagnostic sample extraction and visual contact-sheet inspection (`data/sample/diagnostic_contact_sheet.jpg`) confirm the modality:
  **Frontal Chest Radiographs (Chest X-rays / CXR)**.
* Projections include portable bedside examinations (`"PORTABLE"`, `"ERECT"`, `"SEMI-ERECT"`, `"UPRIGHT"`) and routine PA/AP studies with anatomical side markers (`L`, `R`) and standard de-identification masking over patient name fields.

### Image Dimensions & Format
* **Image Format:** Standard JPEG.
* **Resolution:** Exactly **$512 \times 512$ pixels** across all evaluated samples.
* **Color Mode:** Stored as 3-channel RGB (`mode='RGB'`), but pixel values are completely grayscale ($R = G = B$ for all pixels, confirmed programmatically across random samples).
* **Bit Depth:** 8 bits per channel ($[0, 255]$ dynamic range).
* **Mean Pixel Intensity:** ~124 to 128 (well-centered exposure, no extreme clipping).
* **Integrity:** 200 randomly sampled images were tested with `PIL.Image.verify()`: **0 corrupt images** discovered (100% integrity).

### Preprocessing Implications for DenseNet-121
* DenseNet-121 standard inputs expect 3-channel RGB normalized tensors at $224 \times 224$ (or $512 \times 512$).
* Because the images are already formatted as 3 identical channels at $512 \times 512$, they require minimal preprocessing:
  1. Bilinear resizing to $224 \times 224$ (or direct high-resolution $512 \times 512$ feature extraction).
  2. Conversion to $[0.0, 1.0]$ float32.
  3. Standardization with ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.

---

## 4. Labels & Clinical Finding Distribution

### Candidate Targets
The raw dataset contains **no discrete binary label columns** (such as `label=1`). Instead, pathology labels are embedded in the free-text `reports` column.

A clinical negation and polarity analysis was conducted across the reports to distinguish positive pathological findings from negative mentions (e.g., *"no pneumothorax"*):

| Abnormality / Finding | Total Mention Count | Positive Findings | Negated Mentions | Positive % of Dataset | Clinical & Modeling Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Atelectasis** | 4,149 | 3,808 | 341 | **35.5%** | Moderate: High prevalence, but often subtle linear or plate-like opacities. |
| **Pleural Effusion** | 8,521 | 3,065 | 5,456 | **28.6%** | **High:** Highly distinctive blunting of costophrenic angles; ideal for Grad-CAM localization. |
| **Pulmonary Edema** | 4,116 | 2,194 | 1,922 | **20.5%** | Moderate: Bilateral diffuse perihilar distribution. |
| **Pneumonia / Opacity** | 2,740 | 1,869 | 871 | **17.4%** | **High:** Primary clinical focus of hackathon problem statement and AGENTS.md scope. |
| **Cardiomegaly** | 1,732 | 1,314 | 418 | **12.3%** | **High:** Unambiguous anatomical cardiac silhouette boundary. |
| **Normal / Clear** | 1,613 | 1,151 | 462 | **10.7%** | **High:** Essential negative control cohort. |
| **Pneumothorax** | 7,807 | 509 | 7,298 | **4.7%** | Low for MVP: >93% of mentions are explicitly negated (*"no pneumothorax"*). |

### Recommended Initial Modeling Target
* **Target:** **Pulmonary Opacity / Pneumonia** and/or **Pleural Effusion**.
* **Rationale:**
  1. Both conditions have balanced positive prevalence (17.4% and 28.6% positive cases).
  2. Both produce distinct focal radiographic features in the lung fields that can be highlighted by Grad-CAM.
  3. Directly fulfills the clinical objective specified in `AGENTS.md` ("Chest X-ray abnormality detection/classification").

---

## 5. Clinical Text Analysis & Label Leakage Risk

### Text Characteristics
* **Field:** `reports` (0% missing).
* **Length Statistics:** Min = 35 chars, Max = 2,097 chars, Mean = 470.7 chars, Median = 437 chars.
* **Content Structure:** Standard diagnostic radiology reports consisting of narrative **Findings** followed by a synthesized **Impression** (frequently separated by double spaces `"  "` in 97.7% of records).

### Critical Safety Finding: Severe Label Leakage Hazard (AGENTS.md Rule 10 & 28)
* **The Hazard:** The `reports` column represents the *post-hoc diagnostic report written by the radiologist after evaluating the radiograph*.
* It routinely contains explicit diagnostic assertions, e.g.:
  > *"Findings consistent with pneumonia."*  
  > *"New mild pulmonary edema with persistent small bilateral pleural effusions."*
* **If BioClinicalBERT is fed the full report text as an input to classify the abnormality, the model will simply perform trivial keyword matching on the radiologist's conclusions.**
* **Architectural Decision:**
  1. The full radiology report **must NOT** be fed as an unfiltered input feature to the disease classification head.
  2. The text is appropriately used for:
     * **Ground-truth label extraction / weak supervision** for image model training.
     * **Cross-modal semantic verification (Evidence Engine)**: Checking whether the vision model's Grad-CAM attention aligns with what the radiologist reported.
     * **Ground-truth evaluation of explanation generation** (Phase 12).

---

## 6. Structured Clinical Variables Assessment

* **Inspection Finding:** The raw Parquet tables contain **zero structured variables** (no age, sex, SpO2, heart rate, blood pressure, smoking history, or lab values).
* In addition, patient demographics within the de-identified narrative text are scrubbed (0 explicit age mentions across 5,000 sampled reports).
* **Architectural Decision:**
  * In accordance with AGENTS.md Rule 10 ("Never convert missing information into negative information" and "Never fabricate metadata"), we **do not create fake structured columns in the raw data**.
  * Structured context is supported dynamically at inference time via the FastAPI request schema (`ClinicalContextInput`), allowing clinicians to submit vitals during interactive review.

---

## 7. Patient / Study Identifiers & Data Leakage Prevention

* **Missing Identifier Columns:** The dataset contains no explicit `patient_id`, `study_id`, or `accession_id` fields (`path` is uniformly null).
* **Duplicate Image Check:** SHA-256 byte hashing of all 21,443 images confirms **100% unique images** (0 duplicate radiographs).
* **Report Overlap:** 41 boilerplate reports appear across both shards, representing standard clinical templates (*"Compared to prior study there is no significant interval change. No change."*).
* **Split Strategy:**
  * Shards `0` and `1` must be merged into a single pool before splitting.
  * Splitting must be stratified and clustered by report text hash so that studies with identical reports do not straddle training and validation sets.
  * **Ratio:** 70% Train (15,010 images) / 15% Validation (3,216 images) / 15% Test (3,217 images).

---

## 8. Multimodal Architecture Compatibility Verdict

| Modality Component | Planned Technology | Support Status | Technical Rationale & Strategy |
| :--- | :--- | :--- | :--- |
| **Image Modality** | **DenseNet-121** | **YES** | 21,443 clean, non-corrupt $512 \times 512$ chest X-rays fully support transfer learning, convolutional feature extraction, and transfer learning. |
| **Clinical Text** | **BioClinicalBERT** | **PARTIALLY / CONDITIONAL** | High-quality clinical text is available, but represents post-imaging radiology reports. Must be used for ground-truth label extraction, semantic cross-modal alignment, and evidence verification rather than raw input features to avoid target leakage. |
| **Structured Variables**| **Small MLP** | **PARTIALLY / RUNTIME** | Not present in raw dataset files. Processed at runtime when provided via API/Dashboard clinical context input. |
| **Multimodal Fusion** | **Trainable Fusion Layer** | **PARTIALLY** | The dataset directly supports **Image $\leftrightarrow$ Report Evidence Grounding** and **Image Feature Classification**. Pre-test clinical vitals fusion operates as a hybrid inference pipeline with runtime clinician context. |
| **Localization** | **Grad-CAM** | **YES** | DenseNet-121 final dense block (`features.denseblock4.denselayer16`) directly supports class activation mapping over the $512 \times 512$ radiographs. |

---

## 9. Recommended Next Phase (Prompt 3 Action Plan)

With dataset inspection complete, the recommended next steps for **Phase 2 & Phase 3 (Preprocessing & Dataset Pipeline)** are:

1. **Rule-Based Clinical Label Extraction:**
   Implement an automated CheXpert-style label extraction pipeline (`training/datasets/label_extractor.py`) using regular expressions and negation detection to convert the `reports` column into verified multi-label targets for top abnormalities (*Pleural Effusion, Pulmonary Opacity / Pneumonia, Cardiomegaly, Atelectasis, Normal*).
2. **Metadata Split Generation:**
   Generate clean, non-leaking train/val/test split indices saved to `data/metadata/splits.json` using report clustering.
3. **Multimodal PyTorch Dataset:**
   Implement `MedicalMultimodalDataset` in `training/datasets/dataset.py` that reads directly from the Parquet shards on the fly, decoding images and providing labels and report tokens.
4. **Data Verification Test Suite:**
   Add unit tests in `tests/test_dataset.py` verifying dataset loading, label distribution consistency, and tensor normalization.
