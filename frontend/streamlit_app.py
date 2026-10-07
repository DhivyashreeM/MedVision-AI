"""MedVision AI - Clinical Decision-Support Workstation.

Evidence-grounded multimodal medical AI decision-support platform for chest X-ray review.
Core Principle: Prediction + Localization + Evidence + Confidence
Safety: AI decision-support only. Does NOT replace professional medical judgment.
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="MedVision AI - Clinical Decision Support",
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
}
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 12px; padding: 20px 28px; margin-bottom: 20px;
}
.main-header h1 { margin:0; font-size:1.75rem; font-weight:800; color:#f8fafc; }
.main-header p  { margin:4px 0 0 0; font-size:0.88rem; color:#94a3b8; }
.disclaimer { background:#f0f9ff; border-left:4px solid #0284c7; border-radius:6px;
    padding:11px 16px; font-size:0.85rem; color:#0c4a6e; margin-bottom:18px; }
.demo-badge { background:#fef3c7; border:1px solid #fbbf24; border-radius:6px;
    padding:8px 14px; font-size:0.82rem; font-weight:600; color:#92400e; text-align:center; margin-bottom:14px; }
.section-h { font-size:0.75rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
    color:#64748b; margin:18px 0 8px 0; padding-bottom:5px; border-bottom:1px solid #e2e8f0; }
.finding-card { background:#fff; border:1px solid #e2e8f0; border-radius:10px;
    padding:16px 18px; margin-bottom:12px; box-shadow:0 1px 4px rgba(0,0,0,0.05); }
.finding-card-supported   { border-left:4px solid #0ea5e9; }
.finding-card-conflicting { border-left:4px solid #f59e0b; }
.finding-card-insufficient{ border-left:4px solid #94a3b8; }
.finding-name { font-size:1rem; font-weight:700; color:#0f172a; margin:0 0 4px 0; }
.conf-pct { font-size:1.4rem; font-weight:800; color:#0f172a; }
.conf-level-High     { background:#dbeafe; color:#1e40af; font-size:0.76rem; font-weight:600; padding:2px 9px; border-radius:99px; }
.conf-level-Moderate { background:#fef9c3; color:#854d0e; font-size:0.76rem; font-weight:600; padding:2px 9px; border-radius:99px; }
.conf-level-Low      { background:#f1f5f9; color:#475569; font-size:0.76rem; font-weight:600; padding:2px 9px; border-radius:99px; }
.ev-badge-supported   { display:inline-block; background:#dcfce7; color:#166534; border-radius:6px; padding:4px 11px; font-size:0.78rem; font-weight:700; }
.ev-badge-conflicting { display:inline-block; background:#fef3c7; color:#92400e; border-radius:6px; padding:4px 11px; font-size:0.78rem; font-weight:700; }
.ev-badge-insufficient{ display:inline-block; background:#f1f5f9; color:#475569; border-radius:6px; padding:4px 11px; font-size:0.78rem; font-weight:700; }
.conf-bar-track { background:#e2e8f0; border-radius:99px; height:7px; width:100%; margin:5px 0 3px 0; }
.conf-bar-High     { background:linear-gradient(90deg,#3b82f6,#0ea5e9); border-radius:99px; height:7px; }
.conf-bar-Moderate { background:linear-gradient(90deg,#f59e0b,#fbbf24); border-radius:99px; height:7px; }
.conf-bar-Low      { background:linear-gradient(90deg,#94a3b8,#cbd5e1); border-radius:99px; height:7px; }
.ev-row { display:flex; gap:8px; align-items:flex-start; margin-bottom:5px; font-size:0.83rem; color:#334155; }
.quality-good { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px;
    padding:10px 14px; color:#166534; font-weight:600; font-size:0.86rem; margin-bottom:8px; }
.quality-warn { background:#fffbeb; border:1px solid #fde68a; border-radius:8px;
    padding:10px 14px; color:#92400e; font-weight:600; font-size:0.86rem; margin-bottom:8px; }
.chip { display:inline-block; background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px;
    padding:2px 7px; font-size:0.76rem; font-family:monospace; color:#334155; margin-right:5px; }
.explanation-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
    padding:14px 16px; font-size:0.88rem; color:#334155; line-height:1.6; }
.normal-card { background:#f0fdf4; border:1px solid #86efac; border-radius:10px;
    padding:14px 18px; margin-bottom:12px; }
.status-ok   { color:#16a34a; font-weight:600; }
.status-warn { color:#d97706; font-weight:600; }
.status-err  { color:#dc2626; font-weight:600; }
.limit-item  { font-size:0.82rem; color:#475569; padding:2px 0; }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR   = PROJECT_ROOT / "data" / "sample"
API_URL      = os.getenv("MEDVISION_API_URL", "http://localhost:8000")

SAMPLE_CASES = {
    "None (Upload Custom Image)": None,
    "Case 1 - Suspected Pulmonary Infiltrate": {
        "path": SAMPLE_DIR / "diagnostic_sample_0.jpg",
        "age": 62, "sex": "Female", "spo2": 93.0,
        "symptoms": ["cough", "fever", "shortness of breath"],
        "history": "History of recurrent respiratory infections",
        "notes": "Persistent cough and low-grade fever for 4 days.",
    },
    "Case 2 - Cardiac & Effusion Review": {
        "path": SAMPLE_DIR / "diagnostic_sample_1.jpg",
        "age": 71, "sex": "Male", "spo2": 94.0,
        "symptoms": ["dyspnea", "chest pain"],
        "history": "Congestive heart failure and hypertension",
        "notes": "Bilateral lower extremity edema. Orthopnea on examination.",
    },
    "Case 3 - Pre-operative Clearance": {
        "path": SAMPLE_DIR / "diagnostic_sample_2.jpg",
        "age": 45, "sex": "Female", "spo2": 98.0,
        "symptoms": ["cough"],
        "history": "No known cardiopulmonary disease",
        "notes": "Routine pre-operative clearance examination.",
    },
    "Case 4 - Chest Discomfort Workup": {
        "path": SAMPLE_DIR / "diagnostic_sample_3.jpg",
        "age": 58, "sex": "Male", "spo2": 96.0,
        "symptoms": ["chest pain", "dyspnea"],
        "history": "Remote history of tobacco use",
        "notes": "Exertional chest tightness. ECG non-diagnostic.",
    },
}


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, {}


def run_direct_pipeline(image_bytes, clinical_data=None):
    from app.api.schemas import ClinicalContextInput
    from app.inference.pipeline import InferencePipeline
    pipeline = InferencePipeline.get_instance()
    ctx = ClinicalContextInput(**clinical_data) if clinical_data else None
    resp, heatmaps = pipeline.analyze(image_bytes=image_bytes, clinical_context=ctx)
    return resp.model_dump(mode="json"), heatmaps


def run_analysis(image_bytes, clinical_data=None):
    try:
        files = {"file": ("chest_xray.jpg", image_bytes, "image/jpeg")}
        data  = {"clinical_context": json.dumps(clinical_data)} if clinical_data else {}
        r = requests.post(f"{API_URL}/api/v1/analyze", files=files, data=data, timeout=60)
        if r.status_code == 200:
            rj = r.json()
            return rj, rj.pop("heatmaps", {})
    except Exception:
        pass
    return run_direct_pipeline(image_bytes, clinical_data)


def _conf_bar(pct, level):
    return (f'<div class="conf-bar-track">'
            f'<div class="conf-bar-{level}" style="width:{pct:.1f}%"></div>'
            f'</div>')


def _ev_badge(status):
    m = {
        "supported":    ("\u2713 SUPPORTED",         "ev-badge-supported"),
        "conflicting":  ("\u26a0 EVIDENCE CONFLICT", "ev-badge-conflicting"),
        "insufficient": ("\u25cb INSUFFICIENT",       "ev-badge-insufficient"),
        "not_detected": ("\u25cb NOT DETECTED",       "ev-badge-insufficient"),
    }
    lbl, cls = m.get(status, (status.upper(), "ev-badge-insufficient"))
    return f'<span class="{cls}">{lbl}</span>'


def render_finding_card(f):
    fname  = f.get("finding", "Unknown Finding")
    conf   = f.get("confidence", 0.0)
    pct    = conf * 100
    level  = f.get("confidence_level", "Low")
    status = f.get("evidence_status", "insufficient")
    img_ev = f.get("image_evidence", False)
    cln_ev = f.get("clinical_evidence", False)
    region = f.get("model_attention_region") or "Not localized"
    guide  = f.get("supporting_rationale", "")

    if "no acute" in fname.lower():
        st.markdown(f'''<div class="normal-card">
          <p class="finding-name">\u2713 {fname}</p>
          <p style="font-size:0.83rem;color:#166534;margin:0">{guide}</p>
        </div>''', unsafe_allow_html=True)
        return

    card_cls = {"supported": "finding-card-supported",
                "conflicting": "finding-card-conflicting"}.get(status, "finding-card-insufficient")
    img_icon = "\u2705" if img_ev else "\u2b1c"
    cln_icon = "\u2705" if cln_ev else "\u2b1c"
    img_txt  = "Strong visual attention detected (\u226560%)" if img_ev else "Signal below strong-evidence threshold"
    cln_txt  = "Corroborated by supplied clinical context" if cln_ev else "No corroborating clinical context"

    st.markdown(f'''<div class="finding-card {card_cls}">
      <p class="finding-name">\U0001f50d {fname}</p>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <span class="conf-pct">{pct:.1f}%</span>
        <span class="conf-level-{level}">{level} confidence</span>
        {_ev_badge(status)}
      </div>
      {_conf_bar(pct, level)}
      <p style="font-size:0.76rem;color:#94a3b8;margin:2px 0 10px 0">
        Uncalibrated model score \u00b7 Not a probability of disease
      </p>
      <div class="ev-row"><span>{img_icon}</span>
        <span><strong>Imaging:</strong> {img_txt}</span></div>
      <div class="ev-row"><span>{cln_icon}</span>
        <span><strong>Clinical:</strong> {cln_txt}</span></div>
      <div class="ev-row"><span>\U0001f4cd</span>
        <span><strong>Attention:</strong> {region}</span></div>
      <div style="background:#f8fafc;border-radius:6px;padding:9px 12px;margin-top:10px;
                  font-size:0.84rem;color:#334155;line-height:1.55">
        {guide}
      </div>
    </div>''', unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    st.markdown("### \U0001fa7a MedVision AI")
    st.caption("Evidence-grounded chest X-ray decision support")
    st.divider()

    is_online, _ = check_api_health()
    if is_online:
        st.markdown("**Backend:** <span class='status-ok'>\u25cf Online</span>", unsafe_allow_html=True)
    else:
        st.markdown("**Backend:** <span class='status-warn'>\u25cf Offline (embedded)</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**\U0001f9e0 Model Status**")
    try:
        from app.inference.pipeline import InferencePipeline
        _p = InferencePipeline.get_instance()
        if _p.vision_model.is_trained:
            st.markdown("<span class='status-ok'>\u2713 Finetuned checkpoint loaded</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-warn'>\u26a0 ImageNet backbone only</span>", unsafe_allow_html=True)
        st.caption("DenseNet-121 \u00b7 Grad-CAM \u00b7 Evidence Engine")
    except Exception:
        st.markdown("<span class='status-err'>\u2717 Pipeline unavailable</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**\U0001f4cb Preset Cases**")
    selected_case_name = st.selectbox("Patient case:", list(SAMPLE_CASES.keys()), label_visibility="collapsed")
    case_preset = SAMPLE_CASES[selected_case_name]

    st.markdown("---")
    st.markdown("**\U0001f3af Scope**")
    st.caption(
        "\u2022 Pulmonary Opacity / Pneumonia\n"
        "\u2022 Pleural Effusion\n"
        "\u2022 Cardiomegaly\n"
        "\u2022 Atelectasis\n"
        "\u2022 Normal / No Acute Abnormality"
    )
    st.markdown("---")
    st.caption("Reporting \u226550% \u00b7 Strong evidence \u226560%")


_ck = (selected_case_name
       .replace(" ", "_").replace(":", "").replace("(", "")
       .replace(")", "").replace("-", "").replace("&", ""))

# ---- Header ----
st.markdown('''<div class="main-header">
  <h1>\U0001fa7a MedVision AI</h1>
  <p>Evidence-grounded multimodal medical imaging decision support \u00b7 Hackathon HNX26PSI05</p>
</div>''', unsafe_allow_html=True)

st.markdown('''<div class="disclaimer">
  <strong>\u2695\ufe0f Clinical Advisory:</strong>
  AI-assisted decision support for licensed medical practitioners only.
  This system does <strong>not</strong> produce a confirmed medical diagnosis.
  All findings must be reviewed alongside the original radiograph by a qualified clinician.
</div>''', unsafe_allow_html=True)

if case_preset is not None:
    st.markdown(f'''<div class="demo-badge">
      \U0001f516 DEMO MODE \u2014 Preset: <strong>{selected_case_name}</strong>.
      Results are illustrative only and do not represent validated clinical performance.
    </div>''', unsafe_allow_html=True)

# ---- Input ----
col_img, col_clin = st.columns([1, 1], gap="large")
uploaded_bytes = None
input_image_pil = None

with col_img:
    st.markdown('<div class="section-h">1 \u00b7 Medical Image</div>', unsafe_allow_html=True)
    if case_preset and Path(case_preset["path"]).exists():
        input_image_pil = Image.open(case_preset["path"]).convert("RGB")
        buf = io.BytesIO(); input_image_pil.save(buf, format="JPEG")
        uploaded_bytes = buf.getvalue()
        st.image(input_image_pil, caption=selected_case_name, use_container_width=True)
    else:
        uf = st.file_uploader(
            "Upload frontal chest radiograph (JPEG/PNG):",
            type=["jpg", "jpeg", "png"], key=f"{_ck}_upload",
        )
        if uf:
            uploaded_bytes = uf.read()
            input_image_pil = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
            st.image(input_image_pil, caption="Uploaded radiograph", use_container_width=True)
        else:
            st.info("Upload a chest X-ray or select a preset case from the sidebar.")

with col_clin:
    st.markdown('<div class="section-h">2 \u00b7 Clinical Context (Optional)</div>', unsafe_allow_html=True)
    st.caption("Providing context enables multimodal evidence grounding.")

    with st.expander("Patient Demographics & Vitals", expanded=True):
        age_val  = st.number_input("Age (years):", min_value=1, max_value=120,
                                    value=case_preset["age"] if case_preset else 50, key=f"{_ck}_age")
        sex_opts = ["Unknown", "Male", "Female", "Other"]
        def_sex  = case_preset["sex"] if case_preset else "Unknown"
        sex_val  = st.selectbox("Biological sex:", sex_opts,
                                 index=sex_opts.index(def_sex) if def_sex in sex_opts else 0,
                                 key=f"{_ck}_sex")
        spo2_val = st.slider("SpO\u2082 (%):", 70.0, 100.0,
                              value=float(case_preset["spo2"] if case_preset else 98.0),
                              step=0.5, key=f"{_ck}_spo2")

    with st.expander("Symptoms & History", expanded=True):
        symptom_choices = ["cough", "fever", "shortness of breath", "dyspnea", "chest pain",
                           "hypoxia", "wheezing", "leg swelling", "orthopnea", "hemoptysis"]
        selected_symptoms = st.multiselect(
            "Symptoms:", symptom_choices,
            default=case_preset["symptoms"] if case_preset else [],
            key=f"{_ck}_symptoms",
        )
        history_val = st.text_input("Relevant history:",
                                     value=case_preset["history"] if case_preset else "",
                                     placeholder="e.g. Heart failure, asthma...", key=f"{_ck}_history")
        notes_val = st.text_area("Clinical notes:",
                                  value=case_preset["notes"] if case_preset else "",
                                  placeholder="Findings, auscultation, triage notes...", key=f"{_ck}_notes")

# ---- Analyze button ----
st.markdown("---")
analyze_btn = st.button("\U0001f50d  Analyze Radiograph & Correlate Evidence",
                         type="primary", use_container_width=True)

for k, v in [("mv_result", None), ("mv_heatmaps", {}), ("mv_img_pil", None), ("mv_error", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if analyze_btn:
    if uploaded_bytes is None:
        st.error("Please upload a chest X-ray or select a preset case first.")
    else:
        st.session_state.update({"mv_result": None, "mv_heatmaps": {}, "mv_error": None,
                                  "mv_img_pil": input_image_pil})
        clinical_dict = {
            "age": age_val,
            "sex": sex_val if sex_val != "Unknown" else None,
            "oxygen_saturation": spo2_val,
            "symptoms": selected_symptoms,
            "relevant_history": history_val or None,
            "clinical_notes": notes_val or None,
        }
        steps = [
            "Checking image quality",
            "Running DenseNet-121 vision inference",
            "Generating Grad-CAM attention map",
            "Processing clinical context",
            "Running evidence engine",
            "Preparing clinician summary",
        ]
        prog = st.progress(0, text=f"Step 1/{len(steps)}: {steps[0]}")
        try:
            for i, step in enumerate(steps, 1):
                prog.progress(int(i / len(steps) * 85), text=f"Step {i}/{len(steps)}: {step}")
                time.sleep(0.03)
            result, heatmaps = run_analysis(uploaded_bytes, clinical_dict)
            prog.progress(100, text="\u2713 Analysis complete")
            st.session_state["mv_result"]   = result
            st.session_state["mv_heatmaps"] = heatmaps
        except Exception as exc:
            prog.empty()
            st.session_state["mv_error"] = str(exc)

# ---- Results ----
if st.session_state["mv_error"]:
    st.error("**Analysis Unavailable**")
    with st.expander("Error details"):
        err = st.session_state["mv_error"]
        st.code(err.split("\n")[-1] if "\n" in err else err)
    st.markdown("""**What to do:**
- Verify the image is a valid chest radiograph (JPEG/PNG)
- Check model checkpoint: `models/checkpoints/baseline_densenet121.pth`
- Retry the analysis
""")

elif st.session_state["mv_result"]:
    result   = st.session_state["mv_result"]
    heatmaps = st.session_state["mv_heatmaps"]
    img_pil  = st.session_state["mv_img_pil"]

    findings_list   = result.get("findings", [])
    quality_data    = result.get("quality", {})
    summary_text    = result.get("summary", "")
    recommendations = result.get("recommendations", [])

    st.markdown('<div class="section-h">3 \u00b7 Image Review & AI Assessment</div>', unsafe_allow_html=True)

    active_hm_bytes = None
    active_hm_pil   = None
    overlay_pil     = None
    sel_cam         = None

    if heatmaps:
        avail = list(heatmaps.keys())
        sel_cam = st.selectbox("Attention map for:", avail, key="cam_sel") if len(avail) > 1 else avail[0]
        b64 = heatmaps.get(sel_cam)
        if b64:
            active_hm_bytes = base64.b64decode(b64)
            active_hm_pil   = Image.open(io.BytesIO(active_hm_bytes)).convert("L")
            if img_pil:
                try:
                    from app.explainability.visualization import overlay_attention_heatmap
                    hm_np = np.array(active_hm_pil, dtype=np.float32) / 255.0
                    overlay_pil = overlay_attention_heatmap(img_pil, hm_np, alpha=0.45)
                except Exception:
                    overlay_pil = None

    view_col, assess_col = st.columns([1, 1], gap="large")

    with view_col:
        tab_o, tab_h, tab_ov = st.tabs(["\U0001f5bc Original", "\U0001f525 Attention Map", "\U0001f500 Overlay"])
        with tab_o:
            if img_pil:
                st.image(img_pil, use_container_width=True, caption="Original chest radiograph")
        with tab_h:
            if active_hm_bytes:
                st.image(active_hm_bytes, use_container_width=True, caption=f"Grad-CAM \u2014 {sel_cam}")
                st.caption("\u26a0 Represents model attention. NOT a confirmed lesion boundary.")
            else:
                st.info("No attention map: no finding cleared the strong-evidence threshold (\u226560%).")
        with tab_ov:
            if overlay_pil:
                st.image(overlay_pil, use_container_width=True, caption="Attention overlay")
                st.caption("\u26a0 For illustrative review only.")
            else:
                st.info("Overlay available when attention map is generated.")

    with assess_col:
        is_ok  = quality_data.get("is_acceptable", True)
        issues = quality_data.get("issues", [])
        if is_ok:
            st.markdown('<div class="quality-good">\u2713 Diagnostic quality acceptable</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="quality-warn">\u26a0 Quality: {"; ".join(issues)}</div>', unsafe_allow_html=True)

        dims = quality_data.get("dimensions", [0, 0])
        st.markdown(
            f'<span class="chip">{dims[1]}\u00d7{dims[0]}px</span>'
            f'<span class="chip">Sharpness {quality_data.get("blur_score", 0):.1f}</span>'
            f'<span class="chip">Contrast {quality_data.get("contrast_score", 0):.1f}</span>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-h" style="margin-top:14px">AI Findings</div>', unsafe_allow_html=True)
        if not findings_list:
            st.success("\u2713 No findings met the evidence threshold for AI-assisted reporting.")
            st.caption("This may indicate a normal study or insufficient model confidence. "
                       "Clinical review is always recommended.")
        else:
            for f in findings_list:
                render_finding_card(f)

    if findings_list:
        st.markdown('<div class="section-h">4 \u00b7 Detailed Evidence Analysis</div>', unsafe_allow_html=True)
        for f in findings_list:
            with st.expander(f"\U0001f50d {f.get('finding','Finding')} \u2014 {f.get('evidence_status','').upper()}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Evidence Status**")
                    st.markdown(_ev_badge(f.get("evidence_status", "")), unsafe_allow_html=True)
                    st.markdown("**Image Evidence**")
                    st.markdown("\u2705 Detected (\u226560%)" if f.get("image_evidence") else "\u2b1c Below threshold")
                    st.markdown("**Clinical Evidence**")
                    st.markdown("\u2705 Corroborated" if f.get("clinical_evidence") else "\u2b1c Not provided")
                with c2:
                    conf  = f.get("confidence", 0.0)
                    level = f.get("confidence_level", "Low")
                    st.markdown("**Model Confidence**")
                    st.markdown(f"**{conf*100:.1f}%** \u00b7 {level}")
                    st.markdown(_conf_bar(conf*100, level), unsafe_allow_html=True)
                    st.caption("Uncalibrated model score. Not a clinical probability.")
                    st.markdown("**Attention Region**")
                    st.markdown(f.get("model_attention_region") or "Not localized")
                st.markdown("**Clinician Guidance**")
                st.markdown(f'<div class="explanation-box">{f.get("supporting_rationale","")}</div>',
                            unsafe_allow_html=True)

    st.markdown('<div class="section-h">5 \u00b7 Clinician Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explanation-box">{summary_text}</div>', unsafe_allow_html=True)
    if recommendations:
        st.markdown("**Suggested review steps:**")
        for r in recommendations:
            st.markdown(f"\u2022 {r}")

    recap = {}
    if age_val:                           recap["Age"]      = f"{age_val} years"
    if sex_val and sex_val != "Unknown":  recap["Sex"]      = sex_val
    if spo2_val:                          recap["SpO\u2082"] = f"{spo2_val:.1f}%"
    if selected_symptoms:                 recap["Symptoms"] = ", ".join(selected_symptoms)
    if history_val:                       recap["History"]  = history_val
    if notes_val:                         recap["Notes"]    = notes_val

    if recap:
        with st.expander("\U0001f4cb Clinical Context Supplied", expanded=False):
            for k, v in recap.items():
                st.markdown(f"**{k}:** {v}")
            st.caption("Only explicitly supplied information is used. Missing fields are not assumed negative.")

    with st.expander("\U0001f6e0 Raw Model Response (JSON)", expanded=False):
        st.caption("Full structured API response including all model probabilities and metadata.")
        st.json(result)

    with st.expander("\u26a0 Limitations & Safety Notices", expanded=False):
        for lim in [
            "AI output is decision support only \u2014 not a medical diagnosis.",
            "Grad-CAM highlights model attention, not precise lesion boundaries.",
            "Confidence scores are uncalibrated raw sigmoid outputs.",
            "The evidence engine suppresses findings without adequate supporting evidence.",
            "Performance degrades on images outside the training distribution.",
            "Professional radiologist interpretation of the original study is always required.",
        ]:
            st.markdown(f'<div class="limit-item">\u2022 {lim}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.77rem;padding:6px 0">'
    'MedVision AI \u00b7 HNX26PSI05 \u2014 Multimodal Medical Image Intelligence \u00b7 '
    'Prediction + Localization + Evidence + Confidence \u00b7 '
    '<em>Not an autonomous medical diagnostic device.</em>'
    '</div>',
    unsafe_allow_html=True,
)
