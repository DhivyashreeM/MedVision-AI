"""Evaluation module for MedVision AI vision models.

Calculates standard medical classification metrics:
- ROC-AUC (AUROC) per class and macro-average
- PR-AUC (AUPRC / Average Precision) per class and macro-average
- Precision, Recall / Sensitivity, Specificity, F1-Score
- Accuracy at standard threshold

AGENTS.md Rule 8: Never fabricate metrics. If a model has not been evaluated,
state that it has not been evaluated.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import get_logger
from app.models.vision_model import ChestXrayVisionModel, TARGET_PATHOLOGIES
from training.datasets.dataset import MedicalMultimodalDataset
from training.datasets.split import build_clustered_split
from training.datasets.transforms import get_eval_transform

logger = get_logger("medvision.evaluate")

DEFAULT_PARQUET_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "train-00000-of-00002.parquet",
    PROJECT_ROOT / "data" / "raw" / "train-00001-of-00002.parquet",
]


def evaluate_model(
    model: ChestXrayVisionModel,
    data_loader: DataLoader,
    target_classes: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Run evaluation across data loader and compute full metric suite.

    Returns:
        Dict mapping class_name -> dict of metrics, plus 'macro_avg'.
    """
    model.model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(model.device)
            targets = batch["labels"].numpy()

            logits = model.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_targets.append(targets)
            all_probs.append(probs)

    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    results: Dict[str, Dict[str, Any]] = {}
    auroc_list = []
    auprc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []

    for i, class_name in enumerate(target_classes):
        y_true_col = y_true[:, i]
        y_prob_col = y_prob[:, i]
        y_pred_col = y_pred[:, i]

        metrics = {}

        # AUROC (needs at least one pos and one neg)
        if len(np.unique(y_true_col)) > 1:
            try:
                metrics["auroc"] = round(float(roc_auc_score(y_true_col, y_prob_col)), 4)
                auroc_list.append(metrics["auroc"])
            except Exception:
                metrics["auroc"] = None
        else:
            metrics["auroc"] = None

        # AUPRC (Average Precision)
        if len(np.unique(y_true_col)) > 1:
            try:
                metrics["auprc"] = round(float(average_precision_score(y_true_col, y_prob_col)), 4)
                auprc_list.append(metrics["auprc"])
            except Exception:
                metrics["auprc"] = None
        else:
            metrics["auprc"] = None

        # Precision, Recall / Sensitivity, F1
        prec = precision_score(y_true_col, y_pred_col, zero_division=0)
        rec = recall_score(y_true_col, y_pred_col, zero_division=0)
        f1 = f1_score(y_true_col, y_pred_col, zero_division=0)
        acc = accuracy_score(y_true_col, y_pred_col)

        metrics["precision"] = round(float(prec), 4)
        metrics["recall_sensitivity"] = round(float(rec), 4)
        metrics["f1"] = round(float(f1), 4)
        metrics["accuracy"] = round(float(acc), 4)

        # Specificity: TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_true_col, y_pred_col, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["specificity"] = round(spec, 4)

        metrics["support_positive"] = int(np.sum(y_true_col))
        metrics["support_total"] = len(y_true_col)

        f1_list.append(metrics["f1"])
        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall_sensitivity"])
        specificity_list.append(metrics["specificity"])

        results[class_name] = metrics

    # Macro averages
    macro = {
        "auroc": round(float(np.mean(auroc_list)), 4) if auroc_list else None,
        "auprc": round(float(np.mean(auprc_list)), 4) if auprc_list else None,
        "f1": round(float(np.mean(f1_list)), 4) if f1_list else 0.0,
        "precision": round(float(np.mean(precision_list)), 4) if precision_list else 0.0,
        "recall_sensitivity": round(float(np.mean(recall_list)), 4) if recall_list else 0.0,
        "specificity": round(float(np.mean(specificity_list)), 4) if specificity_list else 0.0,
        "num_classes_evaluated": len(target_classes),
    }
    results["macro_average"] = macro

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedVision AI vision model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, default=200, help="Max test samples to evaluate for speed")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics", help="Directory to save metric JSON")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "vision_metrics.json"

    # Determine checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        default_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "baseline_densenet121.pth"
        if default_ckpt.exists():
            checkpoint_path = str(default_ckpt)

    print("=" * 66)
    print("MEDVISION AI — MODEL EVALUATION ENGINE")
    print("=" * 66)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Status: Not yet evaluated")
        print("Metrics: No trained model checkpoint currently available.")
        print(f"Looked for: {checkpoint_path or 'models/checkpoints/baseline_densenet121.pth'}")
        print("Run training/train_vision.py first to create a trained checkpoint.")
        print("=" * 66)
        # Write honest status file
        with open(metrics_file, "w") as f:
            json.dump({
                "status": "Not yet evaluated",
                "checkpoint": None,
                "reason": "Checkpoint not available",
            }, f, indent=2)
        return 0

    print(f"Loading checkpoint: {checkpoint_path}")
    model = ChestXrayVisionModel(checkpoint_path=checkpoint_path, device=args.device)

    # Check for raw Parquet files
    available_parquets = [p for p in DEFAULT_PARQUET_PATHS if p.exists()]
    if not available_parquets:
        print("Error: No raw Parquet files found in data/raw/")
        return 1

    print(f"Loading {args.split} split from Parquet files...")
    train_idx, val_idx, test_idx = build_clustered_split(available_parquets)
    row_indices = val_idx if args.split == "val" else test_idx
    if args.max_samples:
        row_indices = sorted(row_indices[:args.max_samples])
    else:
        row_indices = sorted(row_indices)

    eval_dataset = MedicalMultimodalDataset(
        parquet_paths=available_parquets,
        indices=row_indices,
        transform=get_eval_transform(),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating {len(eval_dataset)} samples on {args.device}...")
    results = evaluate_model(
        model=model,
        data_loader=eval_loader,
        target_classes=TARGET_PATHOLOGIES,
        threshold=args.threshold,
    )

    # Print summary
    print("\nEVALUATION RESULTS:")
    print("-" * 66)
    print(f"{'Class':<22} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'Sens':<8} {'Spec':<8}")
    print("-" * 66)
    for cls in TARGET_PATHOLOGIES:
        m = results.get(cls, {})
        auroc_str = f"{m.get('auroc', 0.0):.4f}" if m.get("auroc") is not None else "N/A"
        auprc_str = f"{m.get('auprc', 0.0):.4f}" if m.get("auprc") is not None else "N/A"
        f1_str = f"{m.get('f1', 0.0):.4f}"
        sens_str = f"{m.get('recall_sensitivity', 0.0):.4f}"
        spec_str = f"{m.get('specificity', 0.0):.4f}"
        print(f"{cls:<22} {auroc_str:<8} {auprc_str:<8} {f1_str:<8} {sens_str:<8} {spec_str:<8}")
    print("-" * 66)
    macro = results.get("macro_average", {})
    macro_auroc = f"{macro.get('auroc', 0.0):.4f}" if macro.get("auroc") is not None else "N/A"
    macro_auprc = f"{macro.get('auprc', 0.0):.4f}" if macro.get("auprc") is not None else "N/A"
    print(f"{'Macro Average':<22} {macro_auroc:<8} {macro_auprc:<8} {macro.get('f1', 0.0):<8.4f} {macro.get('recall_sensitivity', 0.0):<8.4f} {macro.get('specificity', 0.0):<8.4f}")
    print("=" * 66)

    # Save to metrics file
    report = {
        "status": "Evaluated",
        "checkpoint": checkpoint_path,
        "split": args.split,
        "samples_evaluated": len(eval_dataset),
        "threshold": args.threshold,
        "metrics": results,
    }
    with open(metrics_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
