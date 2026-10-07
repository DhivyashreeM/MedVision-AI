# MedVision AI — Model Architecture & Engineering Plan

## 1. Status Notice

> **Current Status:** The model and dataset have not yet been integrated. This document defines the model architecture, feature extraction interfaces, and training roadmap for Phase 4 through Phase 11.

---

## 2. Vision Model Architecture

### Selected Backbone: DenseNet121
* **Architecture Family**: Dense Convolutional Network (`DenseNet121`).
* **Rationale**:
  1. Dense feature reuse allows effective gradient propagation through shallow and deep representations without vanishing gradients.
  2. Substantially fewer parameters than equivalent-depth ResNet architectures, reducing overfitting on medical cohorts.
  3. Feature map continuity in the final convolutional dense block (`features.denseblock4.denselayer16`) produces stable, anatomically coherent Grad-CAM activation maps.
* **Input Dimensions**: $(B, 3, 224, 224)$
* **Target Abnormalities (14 Standard Chest Classes)**:
  * Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
    Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia.

---

## 3. Clinical Text & Context Architecture (Phase 8)

* **Encoder**: Multi-layer projection network transforming structured clinical tokens (age, oxygen saturation, reported symptoms) into a 128-dimensional embedding.
* **Missingness Preservation**: Explicit masking channels to maintain clinical neutral state for unrecorded patient history.

---

## 4. Multimodal Fusion (Phase 9)

$$\mathbf{h}_{\text{joint}} = \text{ReLU}\left( \mathbf{W}_f [\mathbf{v}_{\text{image}} \,\|\, \mathbf{c}_{\text{clinical}}] + \mathbf{b}_f \right)$$

* **Ablation Protocol**: Multimodal claims require empirical validation against:
  1. Vision-only baseline model
  2. Clinical-context-only baseline model
  3. Joint multimodal fusion model

---

## 5. Explainability Architecture (Phase 6)

* **Method**: Gradient-weighted Class Activation Mapping (Grad-CAM).
* **Target Layer**: `features.denseblock4.denselayer16.conv2`.
* **Clinical Boundary Rule**: All activations are surfaced as **AI Model Attention Regions** for doctor review. Heatmaps are never described as definitive lesion boundaries.
