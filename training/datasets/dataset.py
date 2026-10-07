"""Medical Multimodal Dataset — Full Parquet loader for MedVision AI.

Reflects the actual inspected Parquet dataset schema:
  Columns: ['image', 'reports']
  Image format: 512x512 JPEG binary bytes (struct with 'bytes' and 'path')
  Reports: Narrative clinical radiology reports (Findings + Impression)

Uses lazy row-by-row reading via pyarrow to avoid loading all 500MB into RAM at once.
"""

import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset

from training.datasets.label_extractor import ClinicalLabelExtractor, TARGET_PATHOLOGIES


class MedicalMultimodalDataset(Dataset):
    """
    PyTorch Dataset adapter for the inspected 2-shard Parquet dataset.
    Exposes paired chest radiographs and NLP-extracted binary pathology labels.

    Args:
        parquet_paths: List of paths to the .parquet shard files.
        indices: Optional list of global row indices to include (for train/val/test split).
        transform: Optional torchvision transform applied to PIL images.
        target_classes: List of pathology names to extract labels for.
    """

    def __init__(
        self,
        parquet_paths: List[Path],
        indices: Optional[List[int]] = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_classes: Optional[List[str]] = None,
    ):
        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.transform = transform
        self.target_classes = target_classes or TARGET_PATHOLOGIES
        self.label_extractor = ClinicalLabelExtractor(target_classes=self.target_classes)

        # Build shard boundary index: [(shard_path, local_row_start, num_rows)]
        self._shard_meta: List[Tuple[Path, int, int]] = []
        cumulative = 0
        for path in self.parquet_paths:
            if not path.exists():
                raise FileNotFoundError(f"Parquet shard not found: {path}")
            pf = pq.ParquetFile(path)
            n_rows = pf.metadata.num_rows
            self._shard_meta.append((path, cumulative, n_rows))
            cumulative += n_rows

        self._total_rows = cumulative

        # Apply subset indices if given
        if indices is not None:
            self._indices = indices
        else:
            self._indices = list(range(self._total_rows))

        # Memory-efficient caching of current row group
        self._parquet_files: Dict[Path, pq.ParquetFile] = {}
        self._cached_table: Optional[Any] = None
        self._cached_rg_key: Optional[Tuple[Path, int]] = None

    def __len__(self) -> int:
        return len(self._indices)

    def _locate_row(self, global_idx: int) -> Tuple[Path, int]:
        """Map a global row index to (shard_path, local_row_index)."""
        for shard_path, start, n_rows in self._shard_meta:
            if start <= global_idx < start + n_rows:
                return shard_path, global_idx - start
        raise IndexError(f"Global index {global_idx} out of range (total: {self._total_rows})")

    def _read_row(self, shard_path: Path, local_idx: int) -> Dict[str, Any]:
        """Read a single row from a Parquet shard using row group scanning with cache."""
        if shard_path not in self._parquet_files:
            self._parquet_files[shard_path] = pq.ParquetFile(shard_path)
        pf = self._parquet_files[shard_path]

        remaining = local_idx
        for rg in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            if remaining < rg_rows:
                rg_key = (shard_path, rg)
                if self._cached_rg_key == rg_key and self._cached_table is not None:
                    table = self._cached_table
                else:
                    if self._cached_table is not None:
                        del self._cached_table
                        self._cached_table = None
                        pa.default_memory_pool().release_unused()
                    table = pf.read_row_group(rg, columns=["image", "reports"])
                    self._cached_table = table
                    self._cached_rg_key = rg_key

                row = {col: table[col][remaining].as_py() for col in ["image", "reports"]}
                return row
            remaining -= rg_rows
        raise IndexError(f"Local index {local_idx} exceeds shard row count.")


    @staticmethod
    def decode_image_bytes(image_data: Any) -> Image.Image:
        """Safely convert Hugging Face struct/bytes into PIL RGB Image."""
        if isinstance(image_data, dict):
            raw_bytes = image_data.get("bytes")
        elif isinstance(image_data, (bytes, bytearray)):
            raw_bytes = bytes(image_data)
        else:
            raise ValueError(f"Unexpected image data type: {type(image_data)}")

        if not raw_bytes:
            raise ValueError("Empty image bytes encountered in record.")

        img = Image.open(io.BytesIO(raw_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self._indices[idx]
        shard_path, local_idx = self._locate_row(global_idx)
        row = self._read_row(shard_path, local_idx)

        # Decode image
        image: Image.Image = self.decode_image_bytes(row["image"])

        # Extract labels from radiology report
        report_text: str = row.get("reports", "") or ""
        labels_dict: Dict[str, int] = self.label_extractor.extract_labels(report_text)
        label_vector: List[float] = [float(labels_dict[c]) for c in self.target_classes]

        # Apply image transforms
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        return {
            "image": image_tensor,
            "labels": torch.tensor(label_vector, dtype=torch.float32),
            "report": report_text,
            "global_idx": global_idx,
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Count positive labels across the subset (for class weighting).
        WARNING: This reads all rows sequentially — intended for small subsets or validation.
        """
        counts = {cls: 0 for cls in self.target_classes}
        for idx in range(len(self)):
            item = self[idx]
            labels = item["labels"].tolist()
            for i, cls in enumerate(self.target_classes):
                counts[cls] += int(labels[i])
        return counts


# Alias for backwards compatibility
ChestXrayDatasetInterface = MedicalMultimodalDataset

