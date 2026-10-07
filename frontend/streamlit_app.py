"""MedVision AI — Clinical Decision-Support Workstation.

Evidence-grounded multimodal medical AI decision-support platform for chest X-ray review.
Core Principle: Prediction + Localization + Evidence + Confidence
Safety Principle: AI-assisted decision support only — does not replace professional medical judgment.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import requests
import streamlit as st

# Configure Streamlit page per BRAND_GUIDELINES.md
st.set_page_config(
    page_title="MedVision AI — Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply clinical styling per BRAND_GUIDELINES.md
st.markdown(
    """
    <style>
    /* Clean clinical theme typography & palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .main-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }

    .disclaimer-banner {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        border-right: 1px solid #e2e8f0;
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 24px;
        font-size: 0.88rem;
        color: #334155;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 24px;
        margin-bottom: 14px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .quality-good {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 6px;
        padding: 12px 16px;
        color: #166534;
        font-weight: 500;
    }

    .quality-warning {
        background-color: #fffbeb;
        border: 1px solid #fef08a;
        border-radius: 6px;
        padding: 12px 16px;
        color: #854d0e;
        font-weight: 500;
    }

    .finding-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .finding-badge-mod {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .finding-badge-low {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .evidence-item {
        font-size: 0.85rem;
        padding: 3px 0;
        color: #334155;
    }

    .metric-chip {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-family: monospace;
        color: #334155;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
API_URL = os.getenv("MEDVISION_API_URL", "http://localhost:8000")

# Clinical Sample Case presets
SAMPLE_CASES = {
    "None (Upload Custom Image)": None,
    "Case 1: Suspected Pulmonary Infiltrate (Sample 0)": {
        "path": SAMPLE_DIR / "diagnostic_sample_0.jpg",
        "age": 62,
        "sex": "Female",
        "spo2": 93.0,
        "symptoms": ["cough", "fever", "shortness of breath"],
        "history": "History of recurrent respiratory infections",
        "notes": "Patient presents with persistent cough and low-grade fever for 4 days.",
    },
    "Case 2: Cardiac & Effusion Review (Sample 1)": {
        "path": SAMPLE_DIR / "diagnostic_sample_1.jpg",
        "age": 71,
        "sex": "Male",
        "spo2": 94.0,
        "symptoms": ["dyspnea", "chest pain"],
        "history": "History of congestive heart failure and hypertension",
        "notes": "Bilateral lower extremity edema noted. Orthopnea on examination.",
    },
    "Case 3: Frontal Radiograph Study (Sample 2)": {
        "path": SAMPLE_DIR / "diagnostic_sample_2.jpg",
        "age": 45,
        "sex": "Female",
        "spo2": 98.0,
        "symptoms": ["cough"],
        "history": "No known cardiopulmonary disease",
        "notes": "Routine pre-operative clearance examination.",
    },
    "Case 4: Chest Discomfort Workup (Sample 3)": {
        "path": SAMPLE_DIR / "diagnostic_sample_3.jpg",
        "age": 58,
        "sex": "Male",
        "spo2": 96.0,
        "symptoms": ["chest pain", "dyspnea"],
        "history": "Remote history of tobacco use",
        "notes": "Evaluation for exertional chest tightness. ECG non-diagnostic.",
    },
}


def check_api_health() -> Tuple[bool, Dict[str, Any]]:
    """Check API server availability and return status data."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, {}


def run_direct_pipeline_analysis(
    image_bytes: bytes,
    clinical_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Direct pipeline fallback if FastAPI daemon is not running.
    Guarantees seamless user evaluation without external dependencies.
    """
    from app.api.schemas import ClinicalContextInput
    from app.inference.pipeline import InferencePipeline

    pipeline = InferencePipeline.get_instance()
    context_obj = None
    if clinical_data:
        context_obj = ClinicalContextInput(**clinical_data)

    analysis_response, heatmap_b64_map = pipeline.analyze(
        image_bytes=image_bytes,
        clinical_context=context_obj,
    )
    return analysis_response.model_dump(mode="json"), heatmap_b64_map


def run_api_analysis(
    image_bytes: bytes,
    clinical_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Send analysis request to FastAPI backend with direct fallback."""
    try:
        files = {"file": ("chest_xray.jpg", image_bytes, "image/jpeg")}
        data = {}
        if clinical_data:
            data["clinical_context"] = json.dumps(clinical_data)

        resp = requests.post(f"{API_URL}/api/v1/analyze", files=files, data=data, timeout=30)
        if resp.status_code == 200:
            res_json = resp.json()
            heatmaps = res_json.pop("heatmaps", {})
            return res_json, heatmaps
    except Exception:
        pass
    # Fallback directly to in-process pipeline
    return run_direct_pipeline_analysis(image_bytes, clinical_data)


# ── Header & Disclaimer ───────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 MedVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Evidence-Grounded Multimodal Medical AI Decision Support</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>⚠️ Clinical Advisory Notice:</strong>
        This software provides AI-assisted decision support for licensed medical practitioners.
        It does NOT produce a confirmed medical diagnosis. All findings, attention heatmaps, and confidence scores
        must be reviewed in conjunction with patient history, physical examination, and original radiographic imaging.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Status & Controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🖥️ System Status")
    is_online, health_info = check_api_health()
    if is_online:
        st.success("FastAPI Backend: Online")
        st.caption(f"Target: {API_URL}")
    else:
        st.info("FastAPI Backend: Offline\n(Running in Embedded Pipeline Mode)")

    st.markdown("---")
    st.markdown("### 🧠 Model Architecture")
    st.markdown("**Primary Vision Encoder:** DenseNet-121")
    st.markdown("**Resolution:** 224 × 224 RGB")
    st.markdown("**Explainability:** Grad-CAM on DenseBlock 4")
    st.markdown("**Calibration:** Temperature Scaling (T=1.0)")
    st.markdown("**Decision Rule:** Prediction + Localization + Evidence + Confidence")

    st.markdown("---")
    st.markdown("### 🎯 Target Pathologies")
    st.caption("• Pulmonary Opacity / Pneumonia\n• Pleural Effusion\n• Cardiomegaly\n• Atelectasis\n• Normal / No Acute Abnormality")

    st.markdown("---")
    st.markdown("### 📋 Demo Preset Case")
    selected_case_name = st.selectbox("Select Patient Case:", list(SAMPLE_CASES.keys()))
    case_preset = SAMPLE_CASES[selected_case_name]

# Derive a safe widget key prefix from the selected case name
case_key = selected_case_name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")


# ── Main Input Columns ────────────────────────────────────────────────────────
col_img, col_clin = st.columns([1, 1], gap="medium")

uploaded_bytes: Optional[bytes] = None
input_image_pil: Optional[Image.Image] = None

with col_img:
    st.markdown('<div class="section-header">1. Medical Image Input</div>', unsafe_allow_html=True)

    if case_preset and case_preset["path"].exists():
        input_image_pil = Image.open(case_preset["path"]).convert("RGB")
        buf = io.BytesIO()
        input_image_pil.save(buf, format="JPEG")
        uploaded_bytes = buf.getvalue()
        st.image(input_image_pil, caption=f"Selected Preset: {selected_case_name}", use_container_width=True)
    else:
        uploaded_file = st.file_uploader(
            "Upload Frontal Chest Radiograph (DICOM export, JPEG, or PNG):",
            type=["jpg", "jpeg", "png"],
            help="High-resolution frontal view (PA or AP) recommended.",
        )
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.read()
            input_image_pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
            st.image(input_image_pil, caption="Uploaded Radiograph", use_container_width=True)

with col_clin:
    st.markdown('<div class="section-header">2. Clinical Context (Optional)</div>', unsafe_allow_html=True)
    st.caption("Provide patient presentation to enable multimodal evidence grounding.")

    with st.expander("Patient Demographics & Vitals", expanded=True):
        default_age = case_preset["age"] if case_preset else 50
        age_val = st.number_input(
            "Patient Age (years):", min_value=1, max_value=120, value=default_age,
            key=f"{case_key}_age",
        )

        default_sex = case_preset["sex"] if case_preset else "Unknown"
        sex_options = ["Unknown", "Male", "Female", "Other"]
        sex_idx = sex_options.index(default_sex) if default_sex in sex_options else 0
        sex_val = st.selectbox(
            "Biological Sex:", sex_options, index=sex_idx,
            key=f"{case_key}_sex",
        )

        default_spo2 = case_preset["spo2"] if case_preset else 98.0
        spo2_val = st.slider(
            "Oxygen Saturation (SpO2 %):", min_value=70.0, max_value=100.0,
            value=default_spo2, step=0.5,
            key=f"{case_key}_spo2",
        )

    with st.expander("Symptoms & Clinical History", expanded=True):
        symptom_choices = [
            "cough", "fever", "shortness of breath", "dyspnea",
            "chest pain", "hypoxia", "wheezing", "leg swelling", "orthopnea", "hemoptysis",
        ]
        default_symptoms = case_preset["symptoms"] if case_preset else []
        selected_symptoms = st.multiselect(
            "Reported Symptoms:", symptom_choices, default=default_symptoms,
            key=f"{case_key}_symptoms",
        )

        default_history = case_preset["history"] if case_preset else ""
        history_val = st.text_input(
            "Relevant Medical History:", value=default_history,
            placeholder="e.g. Heart failure, asthma, hypertension",
            key=f"{case_key}_history",
        )

        default_notes = case_preset["notes"] if case_preset else ""
        notes_val = st.text_area(
            "Clinical Notes / Presentation:", value=default_notes,
            placeholder="Document clinical presentation and auscultation findings...",
            key=f"{case_key}_notes",
        )


# ── Action Button ─────────────────────────────────────────────────────────────
st.markdown("---")
analyze_btn = st.button("🔍 Analyze Radiograph & Correlate Evidence", type="primary", use_container_width=True)

if analyze_btn:
    if uploaded_bytes is None:
        st.error("Please upload a medical image or select a preset case to proceed.")
    else:
        with st.spinner("Processing radiograph: Quality screening → DenseNet-121 inference → Grad-CAM attention → Multimodal evidence correlation..."):
            clinical_dict = {
                "age": age_val,
                "sex": sex_val if sex_val != "Unknown" else None,
                "oxygen_saturation": spo2_val,
                "symptoms": selected_symptoms,
                "relevant_history": history_val if history_val else None,
                "clinical_notes": notes_val if notes_val else None,
            }

            result, heatmaps = run_api_analysis(uploaded_bytes, clinical_dict)

        # ── Step 1: Image Quality Assessment Display ──────────────────────────
        st.markdown('<div class="section-header">3. Image Quality Assessment</div>', unsafe_allow_html=True)
        quality_data = result.get("quality", {})
        is_acceptable = quality_data.get("is_acceptable", True)
        issues = quality_data.get("issues", [])
        warning_msg = quality_data.get("warning")

        if is_acceptable:
            st.markdown(
                f"""
                <div class="quality-good">
                    ✓ <strong>Diagnostic Quality Acceptable:</strong> Radiograph meets spatial resolution, contrast, and sharpness standards for AI decision support.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="quality-warning">
                    ⚠️ <strong>Quality Limitation:</strong> {warning_msg or 'Image quality may limit reliable analysis.'}
                </div>
                """,
                unsafe_allow_html=True,
            )

        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        dims = quality_data.get("dimensions", [0, 0, 0])
        q_col1.metric("Dimensions", f"{dims[1]} × {dims[0]} px")
        q_col2.metric("Sharpness Score", f"{quality_data.get('blur_score', 0):.1f}")
        q_col3.metric("Contrast Range", f"{quality_data.get('contrast_score', 0):.1f}")
        q_col4.metric("Mean Brightness", f"{quality_data.get('brightness_score', 0):.1f}")

        # ── Step 2: Tri-View Radiographic Review ──────────────────────────────
        st.markdown('<div class="section-header">4. Visual Localization & Attention Mapping</div>', unsafe_allow_html=True)
        st.caption("Side-by-side comparison of original radiograph, AI attention heatmap (Grad-CAM), and overlay.")

        findings_list = result.get("findings", [])
        active_heatmap_bytes = None
        top_pathology = None

        # Check if heatmaps exist
        if heatmaps:
            # Pick first available heatmap or let user choose
            available_classes = list(heatmaps.keys())
            if len(available_classes) > 1:
                selected_cam_class = st.selectbox("Select Visualized Pathology Attention:", available_classes)
            else:
                selected_cam_class = available_classes[0]

            top_pathology = selected_cam_class
            b64_str = heatmaps.get(selected_cam_class)
            if b64_str:
                active_heatmap_bytes = base64.b64decode(b64_str)

        v_col1, v_col2, v_col3 = st.columns(3)

        with v_col1:
            st.markdown("**Original Chest Radiograph**")
            st.image(input_image_pil, use_container_width=True)

        with v_col2:
            st.markdown("**Model Attention Heatmap (Grad-CAM)**")
            if active_heatmap_bytes:
                st.image(active_heatmap_bytes, use_container_width=True)
            else:
                st.info("No focal abnormality above the strong-evidence threshold to generate attention heatmap.")

        with v_col3:
            st.markdown("**Attention Overlay on Anatomy**")
            if active_heatmap_bytes and input_image_pil:
                # Blend overlay
                from app.explainability.visualization import overlay_attention_heatmap
                hm_pil = Image.open(io.BytesIO(active_heatmap_bytes)).convert("L")
                hm_np = np.array(hm_pil, dtype=np.float32) / 255.0
                blended = overlay_attention_heatmap(input_image_pil, hm_np, alpha=0.45)
                st.image(blended, use_container_width=True)
            else:
                st.info("Overlay available when focal attention region is detected.")

        st.caption("ℹ️ *Notice: Highlighted regions represent DenseNet-121 model attention influencing the prediction — NOT confirmed lesion boundaries.*")

        # ── Step 3: AI-Assisted Findings & Evidence ───────────────────────────
        st.markdown('<div class="section-header">5. Evidence-Grounded AI Findings</div>', unsafe_allow_html=True)

        if not findings_list:
            st.success("✓ **No acute focal abnormalities detected above the diagnostic reporting threshold.**")
        else:
            for idx, finding in enumerate(findings_list):
                fname = finding.get("finding", "Finding")
                conf_level = finding.get("confidence_level", "Moderate")
                conf_val = finding.get("confidence", 0.0)
                status_str = finding.get("evidence_status", "supported")
                region = finding.get("model_attention_region", "Non-localized")
                rationale = finding.get("supporting_rationale", "")
                img_ev = finding.get("image_evidence", False)
                clin_ev = finding.get("clinical_evidence", False)

                badge_class = (
                    "finding-badge-high" if conf_level == "High"
                    else "finding-badge-mod" if conf_level == "Moderate"
                    else "finding-badge-low"
                )

                st.markdown(
                    f"""
                    <div class="card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <h4 style="margin: 0; color: #0f172a;">{fname}</h4>
                            <span class="{badge_class}">Model Confidence: {conf_level} ({conf_val:.0%})</span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong>Attention Region:</strong> {region or 'Diffuse or non-localized'}
                        </div>
                        <div style="background: #f8fafc; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px;">
                            <strong>Evidence Checklist:</strong><br>
                            <span class="evidence-item">{'✓' if img_ev else '○'} <strong>Imaging Evidence:</strong> {'Visual pattern detected on radiograph' if img_ev else 'Not supported by image'}</span><br>
                            <span class="evidence-item">{'✓' if clin_ev else '○'} <strong>Clinical Evidence:</strong> {'Corroborated by symptoms/vitals/history' if clin_ev else 'Clinical context unprovided or non-contributory'}</span><br>
                            <span class="evidence-item">✓ <strong>Localization:</strong> AI attention region verified</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #475569;">
                            <strong>Clinician Guidance:</strong> {rationale}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Step 4: Clinical Summary & Recommendations ───────────────────────
        st.markdown('<div class="section-header">6. Clinician Decision-Support Summary</div>', unsafe_allow_html=True)
        summary_text = result.get("summary", "No summary available.")
        recommendations = result.get("recommendations", [])

        st.info(summary_text)

        if recommendations:
            st.markdown("**Recommended Clinical Next Steps:**")
            for rec in recommendations:
                st.markdown(f"• {rec}")

        # ── Step 5: Full JSON Inspection (collapsible) ────────────────────────
        with st.expander("🛠️ Structured Inspection Payload (JSON)", expanded=False):
            st.json(result)

# ── Footer Safety Disclaimer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; padding: 10px 0;">
        MedVision AI Decision-Support Platform | Built for Hackathon HNX26PSI05 — Multimodal Medical Image Intelligence<br>
        Prediction + Localization + Evidence + Confidence | <em>Not an autonomous medical diagnostic device.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
