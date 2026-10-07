"""Comprehensive dataset validation and statistical audit script for MedVision AI."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.datasets.label_extractor import ClinicalLabelExtractor, TARGET_PATHOLOGIES


PARQUET_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "train-00000-of-00002.parquet",
    PROJECT_ROOT / "data" / "raw" / "train-00001-of-00002.parquet",
]


def audit_dataset():
    print("=" * 65)
    print("MEDVISION AI — DATASET VALIDATION & STATISTICAL AUDIT")
    print("=" * 65)

    extractor = ClinicalLabelExtractor(target_classes=TARGET_PATHOLOGIES)
    total_rows = 0
    report_hashes = set()
    duplicate_reports = 0
    empty_reports = 0

    class_counts = {cls: 0 for cls in TARGET_PATHOLOGIES}
    label_cardinality = []  # Number of positive labels per study

    shard_stats = []

    for path in PARQUET_PATHS:
        if not path.exists():
            print(f"Error: {path} not found.")
            return

        pf = pq.ParquetFile(path)
        num_rows = pf.metadata.num_rows
        total_rows += num_rows
        schema_names = pf.schema.names
        shard_stats.append({
            "path": str(path.name),
            "num_rows": num_rows,
            "columns": schema_names,
            "row_groups": pf.metadata.num_row_groups,
        })
        print(f"Inspecting {path.name}: {num_rows} rows...")

        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=["reports"])
            reports = table["reports"].to_pylist()

            for rep in reports:
                if not rep or len(rep.strip()) == 0:
                    empty_reports += 1
                    continue

                h = hashlib.md5(rep.strip().lower().encode()).hexdigest()
                if h in report_hashes:
                    duplicate_reports += 1
                else:
                    report_hashes.add(h)

                labels = extractor.extract_labels(rep)
                num_pos = sum(labels.values())
                label_cardinality.append(num_pos)

                for cls, val in labels.items():
                    if val == 1:
                        class_counts[cls] += 1

    unique_reports = len(report_hashes)
    cardinality_arr = np.array(label_cardinality)

    print("\nDATASET AUDIT SUMMARY:")
    print(f"Total Rows: {total_rows}")
    print(f"Unique Reports: {unique_reports} ({unique_reports / total_rows:.1%})")
    print(f"Duplicate Reports: {duplicate_reports} ({duplicate_reports / total_rows:.1%})")
    print(f"Empty Reports: {empty_reports}")

    print("\nCLASS DISTRIBUTIONS:")
    print(f"{'Pathology':<24} {'Count':<10} {'Prevalence (%)':<15}")
    print("-" * 50)
    class_dist_list = []
    for cls in TARGET_PATHOLOGIES:
        cnt = class_counts[cls]
        prev = (cnt / total_rows) * 100
        print(f"{cls:<24} {cnt:<10} {prev:.2f}%")
        class_dist_list.append({"pathology": cls, "count": cnt, "prevalence_pct": round(prev, 2)})

    output_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_payload = {
        "dataset_name": "Chest X-ray Diagnostic Parquet Dataset",
        "total_rows": total_rows,
        "unique_reports": unique_reports,
        "duplicate_reports": duplicate_reports,
        "empty_reports": empty_reports,
        "shards": shard_stats,
        "class_counts": class_counts,
        "class_prevalence_pct": {cls: round((class_counts[cls] / total_rows) * 100, 2) for cls in TARGET_PATHOLOGIES},
        "mean_labels_per_study": round(float(np.mean(cardinality_arr)), 2),
        "zero_label_studies_pct": round(float(np.mean(cardinality_arr == 0)) * 100, 2),
        "multi_label_studies_pct": round(float(np.mean(cardinality_arr > 1)) * 100, 2),
    }

    stats_json_path = output_dir / "dataset_statistics.json"
    with open(stats_json_path, "w") as f:
        json.dump(stats_payload, f, indent=2)
    print(f"\nSaved dataset statistics to: {stats_json_path}")

    # Export class distribution CSV
    dist_df = pd.DataFrame(class_dist_list)
    csv_path = output_dir / "class_distribution.csv"
    dist_df.to_csv(csv_path, index=False)
    print(f"Saved class distribution CSV to: {csv_path}")


if __name__ == "__main__":
    audit_dataset()
