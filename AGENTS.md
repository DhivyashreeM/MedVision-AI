# MedVision AI — Agent System Instructions

## 1. Project Identity

Project name: MedVision AI

Project purpose:

Build an evidence-grounded multimodal medical AI decision-support system that assists doctors by analyzing medical images together with available clinical context.

The system is a SECOND-OPINION / DECISION-SUPPORT TOOL.

It is NOT an autonomous diagnostic system and must never present its output as a confirmed medical diagnosis.

Core principle:

> Prediction + Localization + Evidence + Confidence

Critical safety principle:

> No evidence → no asserted finding.

Every reported finding must be traceable to:

1. Evidence in the medical image, OR
2. Evidence in the supplied clinical context, OR
3. Both.

The system must never invent clinical findings, symptoms, history, diagnoses, imaging abnormalities, measurements, or patient information.

---

# 2. Primary Development Goal

Build a working end-to-end prototype suitable for a hackathon demonstration.

The system should eventually support:

1. Medical image upload
2. Image validation
3. Image quality assessment
4. Image preprocessing
5. Vision-model inference
6. Abnormality classification
7. Localization / model attention visualization
8. Clinical-context processing
9. Multimodal fusion
10. Evidence verification
11. Confidence estimation/calibration
12. Evidence-grounded explanation
13. Doctor-oriented recommendation
14. API access
15. Modern medical dashboard
16. Demonstration mode
17. Evaluation and testing

The initial implementation should focus on chest X-ray analysis rather than attempting multiple imaging modalities simultaneously.

Do not expand scope unnecessarily.

---

# 3. Development Philosophy

Build incrementally.

Do NOT attempt to create the entire application in one uncontrolled step.

Follow these stages:

Phase 1 — Project foundation
Phase 2 — Dataset integration
Phase 3 — Data preprocessing
Phase 4 — Vision baseline
Phase 5 — Vision evaluation
Phase 6 — Localization / Grad-CAM
Phase 7 — Image quality assessment
Phase 8 — Clinical context processing
Phase 9 — Multimodal fusion
Phase 10 — Evidence engine
Phase 11 — Confidence calibration
Phase 12 — Explanation generation
Phase 13 — FastAPI backend
Phase 14 — Frontend/dashboard
Phase 15 — Testing
Phase 16 — Demo mode
Phase 17 — Documentation
Phase 18 — Deployment

Only implement the requested phase unless explicitly instructed to proceed further.

---

# 4. Important Development Rule

Before writing significant code:

1. Inspect the existing repository.
2. Inspect existing files.
3. Inspect available datasets.
4. Inspect configuration files.
5. Reuse existing code where appropriate.
6. Do not overwrite working functionality unnecessarily.
7. Do not create duplicate modules.
8. Do not invent dataset structure.
9. Do not assume labels or metadata fields.
10. Do not fabricate model performance.

If information is unknown, inspect the actual files or configuration first.

---

# 5. Dataset Rules

Large datasets must NEVER be committed to Git.

Never put medical datasets inside:

* GitHub
* source-control history
* Docker images
* frontend bundles
* public static folders

Use:

```text
data/
├── raw/
├── processed/
├── metadata/
└── sample/
```

Raw data must be treated as read-only.

Never modify or delete original dataset files.

Create processed copies or derived metadata instead.

Before using a dataset:

1. Inspect its directory structure.
2. Identify image files.
3. Identify labels.
4. Identify patient/study identifiers.
5. Identify reports or clinical notes if available.
6. Identify train/validation/test information.
7. Identify missing values.
8. Identify class imbalance.
9. Check licensing/access restrictions.
10. Document the dataset source.

Never fabricate metadata.

---

# 6. Data Leakage Prevention

Medical datasets must be split carefully.

Prefer patient-level splitting over image-level random splitting whenever patient identifiers are available.

Do not allow images belonging to the same patient to appear across training and validation/test sets.

Never use test data for:

* model training
* threshold selection
* hyperparameter tuning
* calibration
* preprocessing decisions that depend on labels

Document the splitting strategy.

---

# 7. Initial Model Scope

Start with one focused clinical task.

Preferred initial scope:

Chest X-ray abnormality detection/classification.

Use transfer learning rather than training a large CNN from scratch.

Potential architectures include:

* DenseNet
* EfficientNet
* ResNet

Choose the architecture based on:

* dataset
* available compute
* task
* implementation stability
* explainability support

Do not select a model merely because it is popular.

Document the final choice.

---

# 8. Model Training Rules

All training code must be reproducible.

Use:

* fixed/random seeds where appropriate
* explicit configuration files
* versioned preprocessing
* documented hyperparameters
* saved checkpoints
* validation metrics
* test metrics

Track at minimum where applicable:

* accuracy
* precision
* recall/sensitivity
* specificity
* F1
* ROC-AUC
* PR-AUC
* confusion matrix

For medical classification, do not rely on accuracy alone.

Never fabricate metrics.

If a model has not been trained, state that it has not been trained.

If metrics are unavailable, display:

"Not yet evaluated"

rather than inventing numbers.

---

# 9. Explainability Rules

The system must provide visual evidence for image-based findings.

Use Grad-CAM or an equivalent explainability technique where appropriate.

Important:

Grad-CAM represents model attention/importance.

Do NOT describe a Grad-CAM heatmap as a clinically verified lesion segmentation.

Use terminology such as:

* "Model attention region"
* "Region influencing the prediction"
* "AI-highlighted region for review"

Only call something a lesion boundary, segmentation, or exact anatomical measurement when an actual segmentation/localization method and appropriate ground truth support that claim.

---

# 10. Clinical Context Rules

Clinical context may include:

* age
* sex
* symptoms
* oxygen saturation
* relevant history
* laboratory results
* radiology report
* patient notes

Clinical context must be processed separately from image features.

The system must distinguish:

### Observed image evidence

What the vision model detects.

### Clinical evidence

What is present in the supplied patient context.

### Derived assessment

What the system infers after combining the evidence.

Never convert missing information into negative information.

For example:

If smoking history is not provided:

Do NOT state:

"Patient is a non-smoker."

Instead state:

"Smoking history was not provided."

---

# 11. Multimodal Fusion

The architecture should support:

Image encoder
+
Clinical/text encoder
→
Fusion layer
→
Prediction

Possible implementation:

image embedding → image feature vector

clinical text/structured data → clinical feature vector

concatenate/project features

fusion network

prediction head

The fusion layer must be trainable when sufficient data exists.

Do not claim multimodal improvement unless it is experimentally demonstrated.

Compare:

1. Image-only model
2. Clinical/context-only model
3. Multimodal model

Report the results honestly.

---

# 12. Evidence-Grounding Engine

Every final finding must pass an evidence check.

For each finding store:

* finding name
* probability/confidence
* image evidence
* localization information
* clinical evidence
* supporting features
* evidence status

Example:

{
"finding": "Possible pulmonary opacity",
"confidence": 0.81,
"image_evidence": true,
"clinical_evidence": true,
"evidence_status": "supported"
}

Unsupported findings must NOT be presented as established findings.

If evidence is insufficient:

"Insufficient evidence for a reliable AI-assisted assessment."

This is preferable to hallucinating.

---

# 13. LLM Rules

An LLM must NEVER be the primary medical diagnostic engine.

If an LLM is used:

It may only:

* summarize verified model outputs
* explain verified evidence
* organize clinical information
* generate doctor-friendly wording

It must NOT:

* invent findings
* invent patient history
* infer unsupported diagnoses
* override the vision model
* create unsupported confidence
* fabricate evidence

The explanation layer should receive structured verified evidence rather than unrestricted raw patient data whenever possible.

---

# 14. Confidence Rules

Raw softmax probabilities must not automatically be presented as trustworthy clinical confidence.

Where possible, implement calibration such as:

* temperature scaling
* reliability analysis
* Expected Calibration Error
* Brier score

The UI should distinguish:

* High confidence
* Moderate confidence
* Low confidence

Confidence must be described as model confidence, not medical certainty.

Preferred wording:

"AI model confidence: Moderate"

Avoid:

"95% certain the patient has pneumonia."

---

# 15. Image Quality

Before inference, check:

* image readability
* resolution
* dimensions
* file type
* brightness
* contrast
* blur
* noise
* corruption
* unsupported image format

If image quality is inadequate, warn the user.

Example:

"Image quality may limit reliable AI analysis. Please review the original study."

Do not silently produce a high-confidence result from a clearly unusable image.

---

# 16. Safety Language

Never use definitive diagnostic wording such as:

"The patient has pneumonia."

Prefer:

"Possible pulmonary opacity detected. Doctor, consider reviewing the highlighted region."

Use:

"AI-assisted decision support — not a diagnosis."

The system must clearly communicate that final clinical judgment belongs to a qualified healthcare professional.

---

# 17. Privacy

The application must be designed for de-identified data.

Do not display unnecessary:

* names
* addresses
* phone numbers
* medical record numbers
* email addresses
* personally identifiable information

Do not log sensitive patient information unnecessarily.

Avoid storing uploaded images permanently unless explicitly required.

---

# 18. API Rules

Backend technology:

Python + FastAPI.

The API should use structured request/response schemas.

Potential endpoints:

GET /health

GET /model/info

POST /quality-check

POST /analyze

POST /explain

POST /feedback

API responses must be machine-readable JSON.

Errors must be explicit and useful.

Do not expose stack traces to end users.

---

# 19. Frontend Rules

The frontend should be designed for a doctor reviewing an AI-assisted result.

Main flow:

1. Upload image
2. Enter clinical context
3. Run analysis
4. Review result
5. Review evidence
6. Review localization
7. Review confidence
8. Review limitations

Do not create a consumer-style "AI diagnosis" interface.

Avoid alarming red/green medical-status styling.

The system should emphasize:

* evidence
* review
* uncertainty
* clinical oversight

---

# 20. Demo Mode

The final project should include demo cases.

At minimum aim for:

1. Normal/low-abnormality case
2. Supported abnormal case
3. Poor-quality image case
4. Insufficient/conflicting evidence case

Demo data must be clearly identified as demo/de-identified data.

Never claim synthetic/demo cases represent real clinical performance.

---

# 21. Code Quality

Use clean modular Python.

Prefer:

* type hints
* docstrings for important functions
* Pydantic schemas
* configuration files
* logging
* reusable utilities
* unit tests

Avoid:

* giant files
* hard-coded absolute paths
* duplicated code
* hidden global state
* magic numbers
* unnecessary dependencies

Use environment variables for secrets.

Never hard-code API keys.

---

# 22. Dependency Rules

Before adding a dependency:

1. Check whether an existing dependency already provides the capability.
2. Prefer stable and widely used libraries.
3. Avoid unnecessary packages.
4. Update requirements/lock files appropriately.
5. Document important dependencies.

---

# 23. Testing Requirements

Tests should cover:

* image loading
* preprocessing
* dataset parsing
* model inference
* quality checks
* Grad-CAM generation
* clinical feature extraction
* fusion
* evidence validation
* confidence handling
* API endpoints
* invalid input
* missing input
* unsupported file types

The evidence engine must have explicit tests preventing unsupported findings from being presented.

---

# 24. Documentation

Maintain:

README.md

docs/architecture.md

docs/data_pipeline.md

docs/model.md

docs/evaluation.md

docs/safety.md

docs/demo_script.md

Also create, where appropriate:

DATA_CARD.md

MODEL_CARD.md

RESOURCES.md

The documentation must identify:

* datasets
* pretrained models
* open-source libraries
* external APIs
* licenses
* limitations
* evaluation results

---

# 25. Git Rules

Never commit:

* raw datasets
* patient images
* credentials
* API keys
* .env files
* model secrets
* private data
* huge generated files

Use .gitignore appropriately.

---

# 26. Change Management

Before modifying an existing component:

Explain internally what is being changed and why.

Prefer small, reversible changes.

After major changes:

1. Run tests.
2. Check imports.
3. Check syntax.
4. Run relevant pipeline.
5. Update documentation.

Do not leave the project in a broken intermediate state.

---

# 27. Antigravity Agent Behavior

When instructed to implement a phase:

1. Inspect the current project.
2. Determine what already exists.
3. Implement only the requested phase.
4. Reuse existing components.
5. Run relevant checks.
6. Report files created/modified.
7. Report commands executed.
8. Report errors.
9. Report what remains to be done.
10. Do not silently skip failed steps.

Never pretend that training, evaluation, deployment, or testing succeeded if it did not actually run.

---

# 28. Critical Anti-Hallucination Requirement

This project is specifically evaluated on evidence-grounded medical reasoning.

Therefore:

NEVER fabricate:

* diagnoses
* image findings
* patient history
* clinical measurements
* localization
* confidence
* evaluation metrics
* dataset statistics
* model performance

If evidence is unavailable, say so.

If the model is uncertain, expose uncertainty.

If the image is poor, warn the user.

If clinical context conflicts with image evidence, surface the conflict rather than hiding it.

---

# 29. Final Product Principle

The final system should feel like:

"Doctor, consider this finding. Here is where the model is focusing, here is the clinical evidence supporting it, and here is the model's calibrated confidence."

It must NOT feel like:

"AI has diagnosed your patient."

Final product priorities:

1. Evidence
2. Localization
3. Confidence
4. Multimodal reasoning
5. Safety
6. Usability
7. Performance
8. Visual polish
9. Deployment

Never sacrifice evidence-grounding for visual appearance.
Never sacrifice safety for model confidence.
Never sacrifice correctness for a demo.
