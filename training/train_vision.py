"""Full training pipeline for MedVision AI — DenseNet121 chest X-ray classifier.

Usage:
    python training/train_vision.py --config training/configs/baseline.yaml

Trains DenseNet121 on the discovered Parquet dataset using NLP-extracted labels.
Saves checkpoints to models/checkpoints/baseline_densenet121.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger
from app.models.vision_model import DenseNet121Classifier, TARGET_PATHOLOGIES
from training.datasets.dataset import MedicalMultimodalDataset
from training.datasets.split import build_clustered_split
from training.datasets.transforms import get_train_transform, get_eval_transform

logger = get_logger("medvision.train_vision")

PARQUET_PATHS = [
    Path("data/raw/train-00000-of-00002.parquet"),
    Path("data/raw/train-00001-of-00002.parquet"),
]


def compute_pos_weights(train_dataset: MedicalMultimodalDataset, num_classes: int) -> torch.Tensor:
    """
    Compute positive class weights for BCEWithLogitsLoss to handle class imbalance.
    Returns tensor of shape (num_classes,).
    """
    logger.info("Computing class weights from %d training samples (may take a few minutes)...", len(train_dataset))
    counts = np.zeros(num_classes, dtype=np.float32)
    total = len(train_dataset)
    # Sample up to 2000 records for speed
    sample_size = min(2000, total)
    step = max(1, total // sample_size)
    sampled = 0
    for i in range(0, total, step):
        item = train_dataset[i]
        labels = item["labels"].numpy()
        counts += labels
        sampled += 1
    # Avoid division by zero
    counts = np.clip(counts, 1, None)
    neg_counts = sampled - counts
    pos_weights = neg_counts / counts
    logger.info("Class counts (sampled %d): %s", sampled,
                {TARGET_PATHOLOGIES[i]: int(counts[i]) for i in range(num_classes)})
    return torch.tensor(pos_weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if batch_idx % 50 == 0:
            logger.info("Epoch %d | Batch %d/%d | Loss: %.4f",
                        epoch, batch_idx, len(loader), loss.item())
    return total_loss / max(n_batches, 1)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run evaluation, return loss and per-class AUROC estimate."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    val_loss = total_loss / max(n_batches, 1)

    # Per-class AUROC
    auroc_scores = {}
    for i, cls in enumerate(TARGET_PATHOLOGIES):
        y_true = all_labels[:, i]
        y_score = all_logits[:, i]
        if len(np.unique(y_true)) < 2:
            auroc_scores[cls] = float("nan")
        else:
            try:
                auroc_scores[cls] = float(roc_auc_score(y_true, y_score))
            except Exception:
                auroc_scores[cls] = float("nan")

    mean_auroc = float(np.nanmean(list(auroc_scores.values())))
    return {"val_loss": val_loss, "mean_auroc": mean_auroc, **auroc_scores}


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet121 on chest X-ray dataset.")
    parser.add_argument("--config", type=str, default="training/configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for quick testing)")
    args = parser.parse_args()

    # Load YAML config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    epochs = args.epochs or cfg.get("epochs", 10)
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    lr = args.lr or cfg.get("lr", 1e-4)
    device_str = args.device or cfg.get("device", "cpu")
    seed = cfg.get("seed", 42)
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "models/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "baseline_densenet121.pth"

    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 60)
    logger.info("MedVision AI — DenseNet121 Training")
    logger.info("Epochs: %d | Batch: %d | LR: %s | Device: %s", epochs, batch_size, lr, device_str)
    logger.info("=" * 60)

    # Verify dataset files
    for p in PARQUET_PATHS:
        if not p.exists():
            logger.error("Parquet file missing: %s", p)
            sys.exit(1)

    # Build train/val splits
    logger.info("Building report-clustered train/val/test split...")
    train_idx, val_idx, test_idx = build_clustered_split(
        PARQUET_PATHS, train_frac=0.70, val_frac=0.15, seed=seed
    )

    if args.max_samples:
        train_idx = train_idx[:args.max_samples]
        val_idx = val_idx[: args.max_samples // 5]
        logger.info("Limited to max_samples=%d for quick test.", args.max_samples)

    logger.info("Split: train=%d | val=%d | test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Build datasets
    train_ds = MedicalMultimodalDataset(
        parquet_paths=PARQUET_PATHS,
        indices=train_idx,
        transform=get_train_transform(),
    )
    val_ds = MedicalMultimodalDataset(
        parquet_paths=PARQUET_PATHS,
        indices=val_idx,
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Build model
    num_classes = len(TARGET_PATHOLOGIES)
    model = DenseNet121Classifier(num_classes=num_classes, pretrained=True)
    model.to(device)

    # Positive class weights for imbalanced multi-label
    pos_weights = compute_pos_weights(train_ds, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)


    best_auroc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        metrics = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Mean AUROC: %.4f | Time: %.1fs",
            epoch, epochs, train_loss, metrics["val_loss"], metrics["mean_auroc"], elapsed,
        )
        for cls in TARGET_PATHOLOGIES:
            logger.info("  AUROC[%s]: %.4f", cls, metrics.get(cls, float("nan")))

        scheduler.step(metrics["mean_auroc"])

        # Save best checkpoint
        if metrics["mean_auroc"] > best_auroc:
            best_auroc = metrics["mean_auroc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": metrics["val_loss"],
                    "mean_auroc": best_auroc,
                    "target_classes": TARGET_PATHOLOGIES,
                },
                checkpoint_path,
            )
            logger.info("✓ Best checkpoint saved → %s (AUROC: %.4f)", checkpoint_path, best_auroc)

    logger.info("Training complete. Best Val AUROC: %.4f", best_auroc)
    logger.info("Checkpoint: %s", checkpoint_path)
    print(f"\nDone. Best AUROC: {best_auroc:.4f} | Checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
