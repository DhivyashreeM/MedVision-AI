"""Dataset preparation and verification script (Phase 2 Entrypoint)."""

import argparse
import sys
from pathlib import Path
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("medvision.prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Verify and prepare dataset structure.")
    parser.add_argument("--source", type=str, default=None, help="Path to raw dataset")
    args = parser.parse_args()

    settings = get_settings()
    logger.info("MedVision AI — Data Preparation Routine")
    logger.info(
        "Current Status: Phase 1 Foundation. "
        "Dataset downloading and ingestion will occur during Phase 2."
    )

    data_dir = settings.DATA_DIR
    print(f"Data root directory: {data_dir.resolve()}")
    for sub in ["raw", "processed", "metadata", "sample"]:
        p = data_dir / sub
        exists = p.exists()
        print(f" - {sub}/: {'Ready' if exists else 'Missing'}")

    print("\nSTATUS: Datasets have not yet been integrated. Awaiting Phase 2 instructions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
