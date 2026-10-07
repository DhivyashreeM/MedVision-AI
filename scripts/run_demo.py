"""MedVision AI — Demo Launcher and Local Service Orchestrator.

Usage:
    python scripts/run_demo.py --mode cli          # Run end-to-end inference test on sample
    python scripts/run_demo.py --mode api          # Launch FastAPI server on :8000
    python scripts/run_demo.py --mode frontend     # Launch Streamlit dashboard on :8501
    python scripts/run_demo.py --mode all          # Launch both API and Frontend
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli_demo(sample_idx: int = 0):
    """Run full pipeline on a real sample image and print decision-support report."""
    print("=" * 70)
    print("MEDVISION AI — MULTIMODAL DECISION SUPPORT CLI DEMO")
    print("Core Rule: Prediction + Localization + Evidence + Confidence")
    print("=" * 70)

    from PIL import Image
    from app.inference.pipeline import InferencePipeline
    from app.api.schemas import ClinicalContextInput

    sample_img_path = PROJECT_ROOT / "data" / "sample" / f"diagnostic_sample_{sample_idx}.jpg"
    if not sample_img_path.exists():
        print(f"Sample image not found: {sample_img_path}")
        return 1

    print(f"1. Loading Chest Radiograph: {sample_img_path.name}")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    print("2. Providing Simulated Clinical Context:")
    clinical_ctx = ClinicalContextInput(
        age=64,
        sex="female",
        oxygen_saturation=92.5,
        symptoms=["cough", "fever", "shortness of breath"],
        relevant_history="Recurrent lower respiratory infections",
    )
    print(f"   Age: {clinical_ctx.age} | SpO2: {clinical_ctx.oxygen_saturation}% | Symptoms: {clinical_ctx.symptoms}")

    print("3. Executing MedVision AI Inference Pipeline...")
    pipeline = InferencePipeline.get_instance()
    response, heatmaps = pipeline.analyze(
        image_bytes=img_bytes,
        clinical_context=clinical_ctx,
        study_id="DEMO-STUDY-001",
    )

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS & CLINICAL DECISION SUPPORT")
    print("=" * 70)
    print(f"Study ID: {response.study_id}")
    print(f"Image Quality Acceptable: {'YES' if response.quality.is_acceptable else 'NO (Limitation)'}")
    print(f"Image Dimensions: {response.quality.dimensions}")
    print(f"Image Sharpness Score: {response.quality.blur_score}")

    print("\nAssertable AI-Assisted Findings:")
    if not response.findings:
        print("  - No acute focal abnormalities detected above diagnostic threshold.")
    else:
        for idx, f in enumerate(response.findings, 1):
            print(f"  [{idx}] {f.finding}")
            print(f"      Confidence: {f.confidence_level} ({f.confidence:.1%})")
            print(f"      Attention Region: {f.model_attention_region}")
            print(f"      Image Evidence: {'[YES] Confirmed' if f.image_evidence else '[NO] Not seen'}")
            print(f"      Clinical Evidence: {'[YES] Corroborated' if f.clinical_evidence else '[NO] Unprovided'}")
            print(f"      Status: {f.evidence_status}")
            print(f"      Clinician Guidance: {f.supporting_rationale}")

    print(f"\nGenerated Grad-CAM Heatmaps: {list(heatmaps.keys())}")
    print(f"\nClinical Summary:\n{response.summary}")
    print("\nNext Steps Recommended:")
    for rec in response.recommendations:
        print(f"  * {rec}")

    print("\n" + "=" * 70)
    print("Clinical Notice: AI-assisted decision support only. Final diagnosis rests with physician.")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="MedVision AI Demo Runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=["cli", "api", "frontend", "all"],
        help="Demo mode to execute (cli, api, frontend, all)",
    )
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Streamlit port")
    args = parser.parse_args()

    if args.mode == "cli":
        return run_cli_demo(sample_idx=0)

    elif args.mode == "api":
        import uvicorn
        print(f"Starting MedVision AI FastAPI server on port {args.port}...")
        uvicorn.run("app.api.main:app", host="0.0.0.0", port=args.port, reload=False)

    elif args.mode == "frontend":
        print(f"Launching Streamlit dashboard on port {args.frontend_port}...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(PROJECT_ROOT / "frontend" / "streamlit_app.py"),
            "--server.port", str(args.frontend_port),
            "--server.headless", "true",
        ])

    elif args.mode == "all":
        print(f"Starting API server on :{args.port} and Streamlit on :{args.frontend_port}...")
        api_proc = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app.api.main:app",
            "--host", "0.0.0.0", "--port", str(args.port),
        ])
        time.sleep(2)
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                str(PROJECT_ROOT / "frontend" / "streamlit_app.py"),
                "--server.port", str(args.frontend_port),
                "--server.headless", "true",
            ])
        finally:
            api_proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
