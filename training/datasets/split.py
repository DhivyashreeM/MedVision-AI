"""Dataset split utilities for MedVision AI.

Strategy: Report-clustered 70/15/15 split.
Groups identical reports together to reduce patient-level leakage risk.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq


def _report_hash(report: str) -> str:
    """Stable hash for grouping duplicate/near-identical reports."""
    return hashlib.md5(report.strip().lower().encode()).hexdigest()


def build_clustered_split(
    parquet_paths: List[Path],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Build train/val/test index splits using report-text clustering.

    1. Read all reports (no image bytes — lightweight column read).
    2. Group rows by report hash.
    3. Shuffle groups deterministically.
    4. Assign groups to train/val/test to avoid same-report rows spanning splits.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    import random
    rng = random.Random(seed)

    # Collect (global_idx, report_hash)
    hash_to_indices: Dict[str, List[int]] = {}
    global_offset = 0

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=["reports"])
            reports = table["reports"].to_pylist()
            for local_i, report in enumerate(reports):
                h = _report_hash(report or "")
                global_idx = global_offset + local_i
                hash_to_indices.setdefault(h, []).append(global_idx)
            global_offset += len(reports)

    # Shuffle group keys
    group_keys = list(hash_to_indices.keys())
    rng.shuffle(group_keys)

    total = global_offset
    train_target = int(total * train_frac)
    val_target = int(total * val_frac)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for key in group_keys:
        group = hash_to_indices[key]
        if len(train_indices) < train_target:
            train_indices.extend(group)
        elif len(val_indices) < val_target:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    return train_indices, val_indices, test_indices


# Alias for backward compatibility
patient_level_split = build_clustered_split

