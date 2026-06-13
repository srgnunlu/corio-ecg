#!/usr/bin/env python3
"""Download ECGFounder's official PTB-XL target vectors."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

ECGFOUNDER_COMMIT = "04edac702b61c91face519774ddcc0cd712fef23"
LABELS_URL = (
    "https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/"
    f"{ECGFOUNDER_COMMIT}/csv/ptbxl_label.csv"
)
DEFAULT_DESTINATION = Path("data/raw/ptb-xl/ecgfounder_ptbxl_label.csv")
EXPECTED_RECORDS = 21_799
EXPECTED_CLASSES = 150


def validate_labels(path: Path) -> None:
    """Reject incomplete or incompatible upstream label files."""
    labels = pd.read_csv(path, usecols=["filename_hr", "label"])
    if len(labels) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} records, found {len(labels)}")
    if labels.filename_hr.duplicated().any():
        raise ValueError("Official label file contains duplicate filenames")
    if not labels.label.map(lambda value: len(json.loads(value)) == EXPECTED_CLASSES).all():
        raise ValueError("Official label vectors must contain 150 values")


def download_labels(destination: Path = DEFAULT_DESTINATION) -> Path:
    """Download, validate, and atomically install the official label CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(LABELS_URL, temporary_path)
        validate_labels(temporary_path)
        shutil.move(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ECGFounder's official PTB-XL evaluation labels"
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Destination CSV (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()
    destination = download_labels(args.destination)
    print(f"Validated official ECGFounder labels at {destination}")


if __name__ == "__main__":
    main()
