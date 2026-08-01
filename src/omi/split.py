# Patient-grouped, OMI-stratified cross-validation folds for the Chongqing dataset.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_VERSION = 1
DEFAULT_SEED = 20260801
DEFAULT_N_SPLITS = 5

GROUP_COLUMN = "Patient_id"
RECORD_COLUMN = "ecg_row_record"
STRATIFY_COLUMN = "OMI"


@dataclass(frozen=True)
class SplitConfig:
    """Frozen configuration behind a fold assignment."""

    version: int = SPLIT_VERSION
    seed: int = DEFAULT_SEED
    n_splits: int = DEFAULT_N_SPLITS
    group_column: str = GROUP_COLUMN
    record_column: str = RECORD_COLUMN
    stratify_column: str = STRATIFY_COLUMN

    def to_dict(self) -> dict:
        """Serialize for the config YAML."""
        return {
            "version": self.version,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "group_column": self.group_column,
            "record_column": self.record_column,
            "stratify_column": self.stratify_column,
        }


def assign_folds(train_table: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign each training record to one cross-validation fold.

    Grouping is by patient, so every ECG from one patient lands in the same
    fold. 855 training patients carry more than one recording, so an ungrouped
    split would leak the same heart across train and validation.

    Args:
        train_table: Contents of the dataset's train.csv.
        config: Frozen split configuration.

    Returns:
        DataFrame with the record column and its assigned `fold`.
    """
    missing = {config.group_column, config.record_column, config.stratify_column} - set(
        train_table.columns
    )
    if missing:
        raise ValueError(f"train table is missing columns: {sorted(missing)}")

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    folds = pd.Series(-1, index=train_table.index, dtype=int)
    split_iterator = splitter.split(
        train_table,
        y=train_table[config.stratify_column],
        groups=train_table[config.group_column],
    )
    for fold_index, (_, holdout_rows) in enumerate(split_iterator):
        folds.iloc[holdout_rows] = fold_index

    if (folds < 0).any():
        raise ValueError("StratifiedGroupKFold left records unassigned")

    return pd.DataFrame(
        {
            config.record_column: train_table[config.record_column].to_numpy(),
            config.group_column: train_table[config.group_column].to_numpy(),
            "fold": folds.to_numpy(),
        }
    )


def verify_no_patient_leakage(assignment: pd.DataFrame, config: SplitConfig) -> dict:
    """Confirm no patient appears in more than one fold.

    Args:
        assignment: Output of `assign_folds`.
        config: The configuration used to build it.

    Returns:
        Per-fold summary statistics.

    Raises:
        ValueError: If any patient spans multiple folds.
    """
    folds_per_patient = assignment.groupby(config.group_column).fold.nunique()
    leaking = folds_per_patient[folds_per_patient > 1]
    if not leaking.empty:
        raise ValueError(
            f"{len(leaking)} patients span multiple folds — grouping failed"
        )

    return {
        "records": int(len(assignment)),
        "patients": int(assignment[config.group_column].nunique()),
        "folds": int(assignment.fold.nunique()),
    }


def summarise_folds(
    assignment: pd.DataFrame, train_table: pd.DataFrame, config: SplitConfig
) -> pd.DataFrame:
    """Per-fold record, patient and positive-label counts."""
    merged = assignment.merge(
        train_table[[config.record_column, config.stratify_column]],
        on=config.record_column,
        how="left",
    )
    rows = []
    for fold_index, chunk in merged.groupby("fold"):
        positives = int(chunk[config.stratify_column].sum())
        rows.append(
            {
                "fold": int(fold_index),
                "records": int(len(chunk)),
                "patients": int(chunk[config.group_column].nunique()),
                f"{config.stratify_column}_positive": positives,
                f"{config.stratify_column}_rate": positives / len(chunk),
            }
        )
    return pd.DataFrame(rows)


def load_fold_assignment(path: Path) -> pd.DataFrame:
    """Read a frozen fold assignment CSV."""
    return pd.read_csv(path)
