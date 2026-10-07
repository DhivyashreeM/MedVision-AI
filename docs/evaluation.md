# MedVision AI — Model Evaluation & Metric Framework

## 1. Status Notice

> **Current Status:** The model and dataset have not yet been integrated.
> **Evaluation Status:** **Not yet evaluated**
>
> In accordance with AGENTS.md Rule 8 and Rule 28, MedVision AI strictly prohibits fabricating or estimating metrics before empirical model evaluation has taken place.

---

## 2. Evaluation Metric Suite (Planned for Phase 5 & Phase 11)

For medical decision support, accuracy alone is misleading due to severe class imbalance. The evaluation pipeline tracks:

| Metric | Target Clinical Objective |
| :--- | :--- |
| **ROC-AUC** | Area under receiver operating characteristic curve across all 14 target classes |
| **PR-AUC** | Precision-Recall AUC (critical for low-prevalence findings such as Hernia/Pneumothorax) |
| **Sensitivity (Recall)** | Minimizing false negatives on critical acute findings |
| **Specificity** | Minimizing false alarms that cause clinician alert fatigue |
| **F1-Score** | Harmonic balance between precision and recall |
| **Brier Score** | Mean squared difference between predicted probabilities and binary outcomes |
| **Expected Calibration Error (ECE)** | Assessment of probability reliability across confidence bins |

---

## 3. Evaluation Protocol

1. **Independent Test Set**: Patient-isolated test split never seen during training or tuning.
2. **Threshold Calibration**: Decision thresholds determined via validation set optimization (Youden's J statistic or clinical sensitivity preference).
3. **Subgroup Analysis**: Stratified performance evaluation across age brackets and biological sex to identify potential performance disparities.
