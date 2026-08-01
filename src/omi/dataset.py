# Loading for the Chongqing ACS/OMI dataset: labels, raw WFDB signals and median beats.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.wfdb_helpers import read_ecg_signal

# Every binary label the dataset ships. OMI is the primary target; the rest are
# available as auxiliary heads or subgroup keys.
DIAGNOSIS_LABELS: list[str] = ["OMI", "AMI", "STEMI", "NSTEMI", "UA", "CTO", "PCI"]
CULPRIT_LABELS: list[str] = [
    "LM", "PLAD", "MLAD", "DLAD", "DB",
    "PLCX", "MLCX", "DLCX", "OM",
    "PRCA", "MRCA", "DRCA",
]
CONTEXT_LABELS: list[str] = ["VF_VT", "Paced", "Prior_PCI"]

PRIMARY_TARGET = "OMI"

# Median beats are stored as raw interleaved int16 microvolts, 12 leads x 500
# samples (1 s at 500 Hz) — not WFDB, so they need explicit decoding.
MEDIAN_LEAD_COUNT = 12
MEDIAN_SAMPLE_COUNT = 500
_MICROVOLTS_PER_MILLIVOLT = 1000.0


def load_labels(
    data_dir: Path,
    fold_assignment_path: Path | None = None,
    table: str = "train",
) -> pd.DataFrame:
    """Load a label table, optionally joined with the frozen fold assignment.

    Args:
        data_dir: Dataset root containing the CSV/ directory.
        fold_assignment_path: Optional frozen fold CSV; adds a `fold` column.
        table: Either "train" or "test". The test table ships without labels.

    Returns:
        The label table, with `fold` attached when an assignment is given.
    """
    labels = pd.read_csv(data_dir / "CSV" / f"{table}.csv")
    if fold_assignment_path is None:
        return labels

    folds = pd.read_csv(fold_assignment_path)[["ecg_row_record", "fold"]]
    merged = labels.merge(folds, on="ecg_row_record", how="left")
    if merged.fold.isna().any():
        missing = int(merged.fold.isna().sum())
        raise ValueError(f"{missing} records have no fold assignment")
    merged["fold"] = merged.fold.astype(int)
    return merged


def load_raw_signal(data_dir: Path, record_name: str) -> np.ndarray:
    """Load one 10 s recording, preprocessed for ECGFounder.

    The dataset is already 500 Hz with ECGFounder's lead order, so the shared
    reader's reordering and resampling are no-ops here; it still applies the
    z-score normalisation the model expects.

    Args:
        data_dir: Dataset root containing row_data/.
        record_name: Value from `ecg_row_record`, e.g. "04904.dat".

    Returns:
        Array of shape (12, 5000), z-score normalised.
    """
    base = data_dir / "row_data" / record_name.removesuffix(".dat")
    signal, _ = read_ecg_signal(str(base))
    return signal


def load_median_beat(data_dir: Path, record_name: str) -> np.ndarray:
    """Load one median beat and return it in millivolts.

    Args:
        data_dir: Dataset root containing med_data/.
        record_name: Value from `ecg_med_record`, e.g. "04904.med".

    Returns:
        Array of shape (12, 500) in mV.
    """
    path = data_dir / "med_data" / record_name
    samples = np.fromfile(path, dtype="<i2")
    expected = MEDIAN_LEAD_COUNT * MEDIAN_SAMPLE_COUNT
    if samples.size != expected:
        raise ValueError(
            f"{record_name}: expected {expected} int16 samples, got {samples.size}"
        )
    # Interleaved layout: sample-major, one value per lead before advancing.
    return samples.reshape(-1, MEDIAN_LEAD_COUNT).T / _MICROVOLTS_PER_MILLIVOLT


def build_label_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Stack selected label columns into a binary matrix."""
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"table is missing label columns: {sorted(missing)}")
    return table[columns].to_numpy(dtype=np.int64)


def select_fold(
    table: pd.DataFrame, fold: int, *, holdout: bool = True
) -> pd.DataFrame:
    """Return either one fold or everything except it.

    Args:
        table: Label table carrying a `fold` column.
        fold: Fold index.
        holdout: True returns just that fold (validation); False returns the rest.
    """
    if "fold" not in table.columns:
        raise ValueError("table has no fold column — load it with a fold assignment")
    mask = table.fold == fold
    return table[mask if holdout else ~mask].reset_index(drop=True)
