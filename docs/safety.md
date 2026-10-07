# MedVision AI — Clinical Safety, Governance & Ethics

## 1. System Identity & Mission

MedVision AI is an **evidence-grounded medical AI decision-support platform**.

* **Second-Opinion Tool**: The platform is explicitly designed to assist qualified healthcare professionals by providing structured, evidence-grounded review prompts.
* **Non-Autonomous**: The system is **NOT** an autonomous diagnostic device and must never present its conclusions as confirmed medical diagnoses.
* **Final Judgment**: Clinical responsibility and decision-making reside entirely with the attending physician.

> **Status Notice:** The model and dataset have not yet been integrated. This safety protocol forms the governing ethical framework for all subsequent engineering phases.

---

## 2. Core Safety Principles

### 2.1 The Quad-Evidence Doctrine
$$\text{Finding} = \langle \text{Prediction}, \text{Localization}, \text{Evidence}, \text{Confidence} \rangle$$

Every finding presented to a clinician must provide traceable visual attention and verified clinical context.

### 2.2 The Grounding Principle
> **No evidence → no asserted finding.**

Candidate abnormalities lacking corroborating features from either the radiograph or documented clinical history are rejected.

---

## 3. Clinical Language Standards (AGENTS.md Rule 16)

The platform strictly enforces non-definitive, hedged language:

| Prohibited Assertions | Approved Decision-Support Language |
| :--- | :--- |
| "The patient has pneumonia." | *"Possible pulmonary opacity detected. Doctor, consider reviewing the highlighted region."* |
| "Confirmed cardiomegaly." | *"Enlarged cardiac silhouette pattern observed with moderate confidence."* |
| "AI diagnosed..." | *"AI-assisted decision-support finding for clinician review."* |
| "100% certainty / guaranteed" | *"AI model confidence: High (84%)."* |

---

## 4. Anti-Hallucination Constraints for Language Models (Rule 13)

Large Language Models (LLMs) are **never** utilized as diagnostic engines. When deployed for clinician summarization:
1. LLMs receive exclusively structured, verified evidence tokens output by the Evidence Engine.
2. LLMs are prohibited from introducing clinical symptoms, measurements, or conditions not present in the verified input.
3. If evidence is discordant or ambiguous, the summary must explicitly articulate the conflict rather than attempting reconciliation.

---

## 5. Mandatory Clinical Disclaimer

All views, reports, and API responses must prominently feature:

> **Clinical Advisory:** AI-assisted decision support only. This system does not replace professional medical judgment.
