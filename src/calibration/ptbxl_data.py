# Joins saved PTB-XL probability matrices to ECGFounder's official 150-class labels.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.ecg_labels import NUM_CLASSES

GROUND_TRUTH_SOURCE = "PKUDigitalHealth/ECGFounder csv/ptbxl_label.csv"


def load_official_labels(labels_path: Path) -> dict[str, np.ndarray]:
    """Read ECGFounder's official PTB-XL label CSV into filename → target vector.

    Args:
        labels_path: Path to `ecgfounder_ptbxl_label.csv`.

    Returns:
        Mapping from PTB-XL `filename_hr` to a binary 150-length vector.
    """
    labels = pd.read_csv(labels_path, usecols=["filename_hr", "label"])
    if labels.filename_hr.duplicated().any():
        raise ValueError("Official ECGFounder label CSV contains duplicate filenames")

    parsed = {
        str(row.filename_hr): np.asarray(json.loads(row.label), dtype=np.int8)
        for row in labels.itertuples(index=False)
    }
    invalid = [name for name, vector in parsed.items() if len(vector) != NUM_CLASSES]
    if invalid:
        raise ValueError("Official ECGFounder label vectors must contain 150 values")
    return parsed


def load_fold_predictions(
    metrics_dir: Path,
    prefix: str,
    ptbxl_database_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a saved probability matrix and align it with official labels.

    Args:
        metrics_dir: Directory holding `<prefix>_probabilities.npy` and
            `<prefix>_record_ids.npy`.
        prefix: Artefact basename prefix used by the evaluation run.
        ptbxl_database_path: Path to `ptbxl_database.csv`, used to map
            record ids to the `filename_hr` join key.
        labels_path: Path to the official ECGFounder label CSV.

    Returns:
        (y_true, y_prob, provenance) where both matrices cover only the
        records that carry an official label.
    """
    probabilities = np.load(metrics_dir / f"{prefix}_probabilities.npy")
    record_ids = np.load(metrics_dir / f"{prefix}_record_ids.npy")
    if len(probabilities) != len(record_ids):
        raise ValueError(
            f"{prefix}: probability rows ({len(probabilities)}) do not match "
            f"record ids ({len(record_ids)})"
        )

    metadata = pd.read_csv(ptbxl_database_path, index_col="ecg_id")
    filename_by_id = metadata.filename_hr.to_dict()
    parsed_labels = load_official_labels(labels_path)

    matched_rows: list[int] = []
    matched_labels: list[np.ndarray] = []
    for row_index, ecg_id in enumerate(record_ids):
        filename = filename_by_id.get(int(ecg_id))
        if filename is None or filename not in parsed_labels:
            continue
        matched_rows.append(row_index)
        matched_labels.append(parsed_labels[filename])

    if not matched_rows:
        raise ValueError(f"{prefix}: no records could be matched to official labels")

    y_true = np.stack(matched_labels, axis=0).astype(np.int64)
    y_prob = probabilities[matched_rows].astype(np.float64)
    provenance = {
        "prefix": prefix,
        "ground_truth_source": GROUND_TRUTH_SOURCE,
        "records_total": int(len(record_ids)),
        "records_matched": len(matched_rows),
        "records_unmatched": int(len(record_ids)) - len(matched_rows),
    }
    return y_true, y_prob, provenance
