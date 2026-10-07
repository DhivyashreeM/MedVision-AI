# MedVision AI — Hackathon Demonstration Script (2–4 Minutes)

**Hackathon Challenge:** HNX26PSI05 — Multimodal Medical Image Intelligence  
**Role:** Clinical Decision-Support Workstation (Second-Opinion Assistant)  
**Core Motto:** *Prediction + Localization + Evidence + Confidence*  
**Golden Rule:** *No evidence → No asserted finding.*

---

## 1. Quick Launch Commands

To start the live demo:

```bash
# Terminal 1: Launch FastAPI Backend
python scripts/run_demo.py --mode api

# Terminal 2: Launch Streamlit Clinician Workstation
python scripts/run_demo.py --mode frontend
```
*Note: The frontend also includes embedded pipeline fallback, so the workstation functions even if running Streamlit standalone.*

---

## 2. 3-Minute Demo Pitch & Walkthrough

### Part 1: Problem Statement & Introduction (0:00 – 0:45)
- **Speaker:** 
  > *"Good morning judges. In high-volume emergency and radiology departments, clinicians review hundreds of chest radiographs per shift under severe time constraints. Commercial AI systems often present 'black-box' diagnoses saying 'Patient has 95% pneumonia' without showing where or why.*  
  > *MedVision AI takes the opposite approach. We are NOT an autonomous diagnostic tool. We are a doctor-assisting second-opinion engine built on four pillars: Prediction, Localization, Evidence, and Confidence."*
- **Action:** Open the dashboard at `http://localhost:8501`. Point out the persistent Clinical Advisory Banner:
  > *"Clinical Advisory: AI-assisted decision support only. This system does not replace professional medical judgment."*

---

### Part 2: Live Study Analysis & Tri-View Localization (0:45 – 1:45)
- **Action:** In the sidebar or preset picker, select **Case 1: Suspected Pulmonary Infiltrate (Sample 0)**.
- **Show:** The radiograph preview and simulated clinical context (Age 62, Female, SpO2 93%, Cough, Fever, Shortness of breath).
- **Action:** Click **"🔍 Analyze Radiograph & Correlate Evidence"**.
- **Speaker:**
  > *"Notice what happens instantly:*  
  > *1. Image Quality Assessment: First, the system screens spatial resolution, contrast range, and sharpness. If an image is degraded or unreadable, the system warns the doctor rather than blindly producing misleading predictions.*  
  > *2. Tri-View Radiographic Review: Here is the original chest radiograph, the DenseNet-121 Grad-CAM attention heatmap, and the anatomical overlay.*  
  > *We emphasize to clinicians: This is the model attention region influencing the prediction — NOT an asserted lesion segmentation boundary."*

---

### Part 3: Evidence Engine & Anti-Hallucination Safety (1:45 – 2:30)
- **Action:** Scroll down to the **Evidence-Grounded AI Findings** cards.
- **Speaker:**
  > *"Look at the finding card for 'Possible Pleural Effusion'. It displays calibrated confidence: Moderate (72.7%).*  
  > *Crucially, examine the Evidence Checklist:*  
  > *• Imaging Evidence: [✓] Visual pattern detected on radiograph*  
  > *• Clinical Evidence: [✓] Corroborated by shortness of breath and SpO2 below 94%*  
  > *• Localization: [✓] AI attention region localized to mid/lower lung field*  
  > *Now notice: Smoking history was not provided in the intake form. Under our safety rules, MedVision AI NEVER converts missing information into false negative information. It explicitly records: 'smoking_history: Not provided', preventing dangerous clinical assumptions."*

---

### Part 4: Clinician Next Steps & Conclusion (2:30 – 3:00)
- **Action:** Scroll down to **Clinician Decision-Support Summary** and **Recommended Next Steps**.
- **Speaker:**
  > *"The system synthesizes a professional, non-definitive summary:*  
  > *'Doctor, consider reviewing the highlighted region for possible effusion. Clinical correlation is recommended.'*  
  > *To summarize: MedVision AI protects patients and clinicians with verified evidence grounding, transparent localization, calibrated uncertainty, and strict medical guardrails.*  
  > *Thank you. We are ready for your questions."*

---

## 3. Pre-Configured Test Cases Available in Workstation

1. **Case 1 (Sample 0):** Suspected Pulmonary Infiltrate — Age 62, SpO2 93%, fever/cough → Demonstrates multimodal concordance and Grad-CAM localization.
2. **Case 2 (Sample 1):** Cardiac & Effusion Review — Age 71, CHF history, dyspnea → Demonstrates cardiomegaly and effusion attention regions.
3. **Case 3 (Sample 2):** Clear Study — Routine pre-op checkup → Demonstrates 'No Acute Abnormality' low-risk finding.
4. **Case 4 (Sample 3):** Exertional Workup — Chest pain → Demonstrates non-concordant symptom vs imaging review.
5. **Custom Upload:** Any user-uploaded JPEG/PNG radiograph to test live quality check, inference, and Grad-CAM overlay.
