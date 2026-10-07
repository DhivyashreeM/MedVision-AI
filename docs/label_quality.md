# MedVision AI — Clinical Label Extraction Quality Audit

**Document:** `docs/label_quality.md`  
**Standard:** NegBio & CheXpert Negation Detection Protocols  
**Audit Target:** `training/datasets/label_extractor.py`  

---

## 1. Methodology & NLP Architecture

Ground-truth clinical labels in MedVision AI are extracted from narrative free-text radiology reports using regex pattern matching paired with bidirectional negation scope analysis:

1. **Pre-Negation Analysis:** Scans a 6-word window preceding candidate pathology mentions for clinical negation triggers (e.g., *"no evidence of"*, *"without"*, *"free of"*, *"rules out"*, *"no focal"*).
2. **Post-Negation Analysis:** Scans a 4-word window following candidate mentions (e.g., *"is absent"*, *"has resolved"*, *"is unlikely"*).
3. **Target Pathology Taxonomy:**
   - `Pulmonary_Opacity`: Matches *"opacity"*, *"pneumonia"*, *"consolidation"*, *"infiltrate"*, *"airspace disease"*.
   - `Pleural_Effusion`: Matches *"pleural effusion"*, *"blunting of costophrenic angle"*.
   - `Cardiomegaly`: Matches *"cardiomegaly"*, *"enlarged cardiac silhouette"*, *"enlarged heart"*.
   - `Atelectasis`: Matches *"atelectasis"*, *"volume loss"*, *"subsegmental atelectasis"*.
   - `Normal`: Matches *"no acute cardiopulmonary process"*, *"clear lungs"*, *"unremarkable chest"*.

---

## 2. Validation Test Cases & Results

| Case ID | Input Narrative Report Snippet | Expected Labels | Extracted Labels | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | *"Bilateral pleural effusion with cardiomegaly. Lungs clear of focal opacity."* | Effusion=1, Cardiomegaly=1, Opacity=0 | Effusion=1, Cardiomegaly=1, Opacity=0 | **PASS** |
| **TC-02** | *"No evidence of pneumonia, pneumothorax, or pleural effusion."* | Opacity=0, Effusion=0 | Opacity=0, Effusion=0 | **PASS** |
| **TC-03** | *"Clear lungs bilaterally. Heart size is normal. No acute findings."* | Normal=1, All others=0 | Normal=1, All others=0 | **PASS** |
| **TC-04** | *"Subsegmental atelectasis at the left base. Stable cardiac enlargement."* | Atelectasis=1, Cardiomegaly=1 | Atelectasis=1, Cardiomegaly=1 | **PASS** |
| **TC-05** | *"Opacities seen previously have resolved."* | Opacity=0 (Post-negation) | Opacity=0 | **PASS** |

---

## 3. Discovered Dataset Prevalence

Across the 21,443 Parquet records:
- **Pulmonary Opacity:** 43.31% (9,287 positive studies)
- **Atelectasis:** 40.27% (8,636 positive studies)
- **Pleural Effusion:** 34.93% (7,490 positive studies)
- **Cardiomegaly:** 16.64% (3,569 positive studies)
- **Normal Studies:** 11.29% (2,420 studies with clear normal impressions)

---

## 4. Limitations & Known Edge Cases

1. **Uncertainty Phrases:** Statements like *"cannot exclude mild effusion"* or *"possible opacity"* are currently mapped as positive candidate targets in line with CheXpert U-ones policy.
2. **Historical vs. Acute:** Phrases such as *"prior granulomatous disease"* do not trigger acute opacity patterns, avoiding false positive acute labeling.
3. **No Direct Structured Metadata:** All training labels depend on the extracted report text; therefore, clinical reports must remain strictly isolated from input feature vectors during vision model training to prevent direct label leakage.
