# MedVision AI — Brand & Product Design Guidelines

## 1. Brand

Product name:

MedVision AI

Product category:

Evidence-grounded multimodal medical AI decision-support platform.

Brand personality:

* Clinical
* Trustworthy
* Calm
* Precise
* Modern
* Professional
* Human-centered
* Evidence-driven

The product should feel like professional clinical software, not a consumer AI chatbot.

---

# 2. Core Brand Message

Primary message:

> Evidence-grounded AI assistance for medical image review.

Supporting message:

> Prediction + Localization + Evidence + Confidence

The product helps clinicians review medical images and clinical context.

It does not replace clinical judgment.

---

# 3. UX Principle

The interface should continuously answer four questions:

1. What did the AI detect?
2. Where is it?
3. Why does the AI think this?
4. How confident is it?

The user should never need to guess where a conclusion came from.

---

# 4. Visual Style

Use a modern clinical dashboard aesthetic.

Preferred characteristics:

* clean
* spacious
* minimal
* professional
* high readability
* subtle borders
* moderate corner rounding
* clear information hierarchy
* restrained use of visual effects

Avoid:

* flashy gradients everywhere
* excessive animations
* gaming aesthetics
* neon colors
* cartoon illustrations
* excessive glassmorphism
* cluttered dashboards
* excessive shadows

---

# 5. Color Philosophy

Use a restrained medical interface palette.

Primary UI:

* white
* off-white
* neutral gray
* dark charcoal
* restrained blue/teal accent

Use status colors only where semantically appropriate.

Confidence should not rely exclusively on color.

For example:

High confidence:
text + icon + visual indicator

Moderate confidence:
text + icon + visual indicator

Low confidence:
text + icon + visual indicator

Never communicate medical certainty using color alone.

---

# 6. Typography

Use a highly readable modern sans-serif font.

Recommended choices:

* Inter
* IBM Plex Sans
* Source Sans 3

Use clear hierarchy:

Page title
→ section title
→ card title
→ body
→ supporting text

Do not use decorative fonts.

---

# 7. Dashboard Layout

Preferred structure:

---

## Header

## Patient / Study Context

## Image Review

## Original Image | AI Attention / Heatmap

## AI Assessment

## Finding | Confidence | Evidence

## Clinical Context

## Explanation / Recommendation

## Limitations / Safety Notice

The image should remain the visual center of the analysis screen.

---

# 8. Image Viewer

The image viewer must support:

* original image
* AI attention heatmap
* overlay view
* zoom
* fit-to-screen
* side-by-side comparison

Clearly label each visualization.

Examples:

"Original X-ray"

"AI attention map"

"AI attention overlay"

Do not label Grad-CAM as:

"Exact lesion"

unless a true lesion segmentation model supports that claim.

---

# 9. Findings Card

Each finding should show:

Finding name

Confidence

Evidence status

Localization

Clinical support

Example:

Possible pulmonary opacity

AI model confidence
Moderate

Image evidence
Supported

Clinical evidence
Supported

Review highlighted region

---

# 10. Evidence Visualization

Evidence should be visually obvious.

Use sections such as:

### Image Evidence

"Model attention concentrated in the right lower lung region."

### Clinical Evidence

"Patient context includes cough and reduced oxygen saturation."

### Evidence Status

"Supported by image + clinical context."

The interface should make it difficult to confuse evidence with speculation.

---

# 11. Confidence Design

Use:

High
Moderate
Low

Also display numerical probability when appropriate.

Example:

Model confidence
82%

Confidence should always be described as AI/model confidence.

Avoid:

"82% diagnosis certainty"

Use:

"AI model confidence: 82%"

---

# 12. Clinical Language

Use respectful clinical language.

Preferred:

"Possible finding"

"AI-assisted assessment"

"Consider reviewing"

"Model attention region"

"Evidence supporting the finding"

"Insufficient evidence"

Avoid:

"Definitely"

"Confirmed"

"The patient has..."

"AI diagnosed..."

"Guaranteed"

---

# 13. Primary Action

The main action should be obvious.

Example:

[ Analyze Image ]

Secondary actions:

[ View Evidence ]

[ Compare ]

[ Review Explanation ]

[ Reset ]

Do not make the interface look like the AI is making an irreversible clinical decision.

---

# 14. Upload Experience

The upload area should clearly communicate:

Supported image formats

Maximum file size

Image quality expectations

Privacy/de-identification reminder

Example:

"Upload a de-identified chest X-ray for AI-assisted review."

---

# 15. Poor Quality State

When an image is poor quality, the UI should clearly communicate the limitation.

Example:

### Image Quality Warning

"Image quality may limit reliable AI analysis."

The system should not display a confident clinical finding without making the quality limitation visible.

---

# 16. Insufficient Evidence State

Use a calm and explicit state.

Example:

### Insufficient Evidence

"The available image and clinical context do not provide sufficient evidence for a reliable AI-assisted finding."

Avoid making this look like a system failure.

It is a safety feature.

---

# 17. Conflict State

If image evidence and clinical context disagree:

### Evidence Conflict

"Image and clinical context provide conflicting signals. Review the original study and clinical information."

Never hide the conflict.

---

# 18. Disclaimer

The application should display a persistent but unobtrusive disclaimer:

"AI-assisted decision support only. This system does not replace professional medical judgment."

Do not use frightening warning banners.

---

# 19. Navigation

Recommended navigation:

Dashboard
Analysis
History
Model Information
Evaluation
About

For a hackathon MVP, Dashboard + Analysis + Model Information may be sufficient.

---

# 20. Model Information Page

Show:

Model architecture

Dataset

Training scope

Supported finding(s)

Evaluation metrics

Limitations

Version

Last updated

This increases transparency.

---

# 21. Loading States

Use meaningful loading messages.

Instead of:

"Loading..."

Use:

"Preparing image..."

"Assessing image quality..."

"Running image analysis..."

"Generating attention map..."

"Combining clinical context..."

"Validating evidence..."

"Preparing clinician review..."

Do not imply that the model is reasoning like a human.

---

# 22. Error Messages

Errors must be understandable.

Bad:

"500 Internal Server Error"

Better:

"We couldn't process this image. Please verify that the file is a supported medical image format."

Technical details should be available to developers through logs, not exposed unnecessarily to clinicians.

---

# 23. Responsive Design

The application should work on:

* laptop
* desktop
* tablet

Prioritize desktop/laptop because medical image review requires sufficient screen space.

---

# 24. Accessibility

Support:

* keyboard navigation
* readable contrast
* descriptive labels
* accessible buttons
* non-color-only status indicators
* clear error messages

Medical information must remain readable.

---

# 25. Animations

Animations should be subtle.

Use animation only to communicate:

* loading
* state changes
* transitions

Avoid decorative animations.

---

# 26. Product Voice

The system should speak like a careful clinical assistant.

Example:

"Doctor, consider reviewing the highlighted region. The model identified a possible pulmonary opacity with moderate confidence."

Not:

"We found something!"

Not:

"Your patient definitely has pneumonia!"

---

# 27. Brand Rule

Every screen should reinforce:

Evidence over assertion.

The product should visually communicate:

"I can show you why this prediction was made."

not:

"Trust the AI."

---

# 28. Hackathon Demo Principle

The demo should communicate the complete story within a few minutes:

1. Upload X-ray
2. Add patient context
3. Analyze
4. Show prediction
5. Show localization
6. Show confidence
7. Show clinical evidence
8. Show explanation
9. Show safety/limitations

The interface should make this flow visually obvious.

---

# 29. Final Design Statement

MedVision AI should look like:

A trustworthy clinical decision-support workstation.

It should NOT look like:

A generic AI chatbot, consumer health app, or flashy AI demo.
