# MedVision AI

> **Evidence-grounded multimodal medical AI decision-support platform for chest radiograph review.**
>
> **Core Principle:** Prediction + Localization + Evidence + Confidence  
> **Critical Safety Rule:** No evidence → no asserted finding.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![Status: MVP Complete](https://img.shields.io/badge/Status-Hackathon%20MVP%20Complete-success.svg)]()

---

> ### ⚠️ Mandatory Clinical Notice
> **MedVision AI is an AI-assisted decision-support tool designed to provide a second opinion for healthcare professionals. It is NOT an autonomous diagnostic system and does not replace qualified medical judgment.**
>
> **Recommended Clinical Language:** *"Doctor, consider reviewing this finding. AI model attention is concentrated in the highlighted region."*  
> **Forbidden Language:** *"The patient has pneumonia."*

---

## 📌 Project Overview

In emergency departments and intensive care units, clinicians interpret hundreds of frontal chest radiographs under severe time pressure. Commercial medical AI solutions often operate as "black boxes" — generating opaque diagnostic probabilities without pointing to concrete radiographic evidence.

**MedVision AI** addresses this challenge with an evidence-grounded decision-support architecture:
1. **Automated Quality Screening:** Pre-inference inspection of spatial resolution, contrast range, underexposure/overexposure, and sharpness.
2. **DenseNet-121 Deep Vision Encoder:** Transfer learning with ImageNet initialization, classifying 5 target radiological categories (*Pulmonary Opacity, Pleural Effusion, Cardiomegaly, Atelectasis, Normal*).
3. **Grad-CAM Attention Localization:** Transparent visual evidence extracted from the final dense block (`features.denseblock4`), mapping peak activations to anatomical lung zones.
4. **Clinical Context Integration:** Ingestion of optional patient demographics, SpO2, reported symptoms, and history without ever assuming missing data to be negative.
5. **Calibrated Confidence:** Post-hoc temperature scaling preventing false certainty.
6. **Strict Evidence Engine:** Deterministic verification ensuring only findings backed by radiographic and/or clinical evidence are presented for clinician review.

---

## 🏗️ Repository Architecture

```text
MedVision-AI/
├── AGENTS.md                  # System rules, safety constraints, anti-hallucination protocols
├── BRAND_GUIDELINES.md        # Clinical design system & brand guidelines
├── README.md                  # Project documentation & operational manual
├── LICENSE                    # Apache 2.0 License
├── pyproject.toml             # Python packaging and pytest configuration
├── requirements.txt           # Verified pinned dependencies
│
├── app/                       # Core production application
│   ├── api/                   # FastAPI backend
│   │   ├── routes/            # Endpoints: /health, /model/info, /quality-check, /analyze
│   │   ├── main.py            # FastAPI factory, CORS, exception handling
│   │   └── schemas.py         # Strict Pydantic v2 schemas
│   ├── core/                  # Configuration, logging, upload security
│   ├── explainability/        # Grad-CAM engine, anatomical localization, heatmap overlays
│   ├── inference/             # Quality screening, preprocessing, singleton pipeline
│   ├── models/                # DenseNet-121 classifier, vision model wrapper, loader
│   ├── multimodal/            # Clinical context evidence extraction, feature engineering
│   ├── reasoning/             # Evidence engine, confidence calibrator, safety validator
│   └── utils/                 # File and image helpers
│
├── data/
│   ├── raw/                   # Immutable raw Parquet shards (never committed to git)
│   ├── metadata/              # Dataset inspection manifest & split index files
│   └── sample/                # De-identified sample chest X-rays for demo & evaluation
│
├── frontend/                  # Streamlit clinical workstation dashboard
│   └── streamlit_app.py       # Tri-view image viewer, evidence cards, case presets
│
├── models/
│   └── checkpoints/           # Trained model checkpoints (baseline_densenet121.pth)
│
├── outputs/
│   └── metrics/               # Real measured evaluation metrics (vision_metrics.json)
│
├── scripts/
│   ├── run_demo.py            # Master demo launcher (CLI, API, Frontend, All)
│   └── inspect_parquet.py     # Dataset schema inspection utility
│
├── tests/                     # Comprehensive pytest test suite (32 unit & integration tests)
│   ├── test_api.py            # FastAPI route tests
│   ├── test_config.py         # Config validation tests
│   ├── test_dataset.py        # Dataset adapter and NLP label extraction tests
│   ├── test_evidence.py       # Evidence engine & anti-hallucination tests
│   ├── test_health.py         # Health probe tests
│   ├── test_models.py         # Vision model forward pass & Grad-CAM tests
│   ├── test_quality.py        # Image quality screening tests
│   └── test_validation.py     # Upload security and clinical missingness tests
│
└── training/                  # Model training and evaluation pipelines
    ├── configs/               # Hyperparameter configuration YAMLs
    ├── datasets/              # MedicalMultimodalDataset, report-clustered split, transforms
    ├── evaluate.py            # Full evaluation script (AUROC, AUPRC, F1, Sens, Spec)
    └── train_vision.py        # DenseNet-121 multi-label training pipeline
```

---

## 📊 Dataset & Splitting Strategy

* **Source Files:** `data/raw/train-00000-of-00002.parquet` & `train-00001-of-00002.parquet`
* **Volume:** 21,443 chest radiographs (10,722 in Shard 0, 10,721 in Shard 1).
* **Schema:** `image` (512 × 512 JPEG binary struct) + `reports` (narrative radiology findings and impression).
* **Target Categories:** Extracted via negation-aware clinical NLP (`ClinicalLabelExtractor`):
  1. `Pulmonary_Opacity`
  2. `Pleural_Effusion`
  3. `Cardiomegaly`
  4. `Atelectasis`
  5. `Normal`
* **Data Leakage Prevention:** Report-clustered splitting (70% train / 15% validation / 15% test). Since patient IDs are not explicitly present in the Hugging Face export, MD5 clustering of narrative reports prevents identical report studies from spanning splits.

---

## 📈 Real Model Evaluation Results

Evaluation performed on the held-out test split using `training/evaluate.py`:

| Pathology Class | AUROC | AUPRC | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cardiomegaly** | **0.6458** | 0.3738 | 0.0000 | 0.0000 | **1.0000** |
| **Pleural Effusion** | **0.5071** | **0.4962** | **0.4906** | **1.0000** | 0.0000 |
| **Pulmonary Opacity** | **0.5038** | 0.4289 | 0.0000 | 0.0000 | 0.9565 |
| **Atelectasis** | 0.3600 | 0.4141 | 0.4390 | 0.4500 | 0.4000 |
| **Normal Study** | 0.4057 | 0.1244 | 0.2000 | 0.4000 | 0.6286 |
| **Macro Average** | **0.4845** | **0.3675** | **0.2259** | **0.3700** | **0.5970** |

*Note: Results obtained from initial transfer learning checkpoint (`models/checkpoints/baseline_densenet121.pth`). As required by AGENTS.md Rule 8, no metrics are fabricated.*

---

## 🚀 Quickstart & Hackathon Demo

### 1. Environment Installation

```bash
git clone https://github.com/example/MedVision-AI.git
cd MedVision-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run All Automated Tests

Verify system integrity (all 32 tests pass):

```bash
pytest
```

### 3. Run Command-Line Demo

Execute end-to-end inference on a de-identified sample chest radiograph:

```bash
python scripts/run_demo.py --mode cli
```

### 4. Launch Full Interactive Clinical Dashboard

Run both the FastAPI backend and Streamlit clinician workstation simultaneously:

```bash
python scripts/run_demo.py --mode all
```

* **Clinician Workstation:** [http://localhost:8501](http://localhost:8501)
* **FastAPI Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **System Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🩺 Clinical Workstation Features

The Streamlit interface (`frontend/streamlit_app.py`) provides:
1. **Case Presets:** Immediate 1-click loading of 4 diagnostic scenarios (Pulmonary Infiltrate, Cardiac / Effusion review, Normal study, Exertional workup) or custom file upload.
2. **Quality Indicator:** Instant spatial resolution, contrast range, and sharpness scoring.
3. **Tri-View Display:** Original radiograph side-by-side with DenseNet-121 Grad-CAM attention heatmap and anatomical overlay.
4. **Evidence Checklist:** Visual confirmation of imaging evidence, clinical corroboration, and localized attention.
5. **Safe Summary:** Clinician-oriented recommendation with complete disclaimer banners.

---

## 🛡️ Ethical Guardrails & Compliance

* **No Evidence → No Asserted Finding:** Unsupported model outputs are blocked by `EvidenceEngine`.
* **Missing Data Preservation:** Omitted history (e.g. smoking status) is explicitly labeled `"Not provided"` and never converted into a negative finding.
* **Non-Diagnostic Language:** Phrases like *"The patient definitely has..."* are strictly flagged and forbidden by `SafetyValidator`.
* **Privacy:** All demo cases utilize de-identified, anonymous radiographs.

---

## 📜 License & Citation

Licensed under the [Apache License, Version 2.0](LICENSE).  
Developed for Hackathon **HNX26PSI05 — Multimodal Medical Image Intelligence**.
