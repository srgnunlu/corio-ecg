#!/usr/bin/env python3
# Build and freeze the patient-grouped OMI cross-validation folds.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.omi.split import (
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    SplitConfig,
    assign_folds,
    summarise_folds,
    verify_no_patient_leakage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_ASSIGNMENT_PATH = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "omi" / "omi_split_v1.yaml"


def main() -> None:
    """CLI entry point for freezing the OMI fold assignment."""
    parser = argparse.ArgumentParser(
        description="Freeze patient-grouped, OMI-stratified CV folds"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--assignment-out", type=Path, default=DEFAULT_ASSIGNMENT_PATH)
    parser.add_argument("--config-out", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing frozen assignment (breaks comparability)",
    )
    args = parser.parse_args()

    if args.assignment_out.exists() and not args.force:
        raise SystemExit(
            f"{args.assignment_out} already exists. The split is frozen so that every "
            "experiment is comparable — pass --force only if you intend to invalidate "
            "all previous results."
        )

    config = SplitConfig(seed=args.seed, n_splits=args.n_splits)
    train_table = pd.read_csv(args.data_dir / "CSV" / "train.csv")
    print(f"Loaded {len(train_table)} training records")

    assignment = assign_folds(train_table, config)
    stats = verify_no_patient_leakage(assignment, config)
    print(
        f"No patient leakage: {stats['records']} records / "
        f"{stats['patients']} patients across {stats['folds']} folds"
    )

    summary = summarise_folds(assignment, train_table, config)
    print("\n" + summary.to_string(index=False))
    print(
        f"\nOMI rate spread across folds: "
        f"{summary.OMI_rate.min():.4f} – {summary.OMI_rate.max():.4f}"
    )

    args.assignment_out.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.assignment_out, index=False)
    with open(args.config_out, "w") as handle:
        yaml.safe_dump(
            {
                **config.to_dict(),
                "source_table": "CSV/train.csv",
                "records": stats["records"],
                "patients": stats["patients"],
                "assignment_file": args.assignment_out.name,
                "status": "frozen",
                "note": (
                    "Fold 0 is the primary validation fold for model and threshold "
                    "selection. The dataset's own test.csv stays untouched and is "
                    "scored only through the authors' online platform."
                ),
            },
            handle,
            sort_keys=False,
        )
    print(f"\nFrozen assignment → {args.assignment_out}")
    print(f"Frozen config     → {args.config_out}")


if __name__ == "__main__":
    main()
