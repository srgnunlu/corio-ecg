#!/usr/bin/env python3
# Fits per-class calibration on the PTB-XL validation fold and writes the artefact.

from __future__ import annotations

import argparse
from pathlib import Path

from src.calibration.fit import fit_calibration
from src.calibration.ptbxl_data import load_fold_predictions
from src.calibration.tiers import (
    MINIMUM_CALIBRATION_POSITIVES,
    TIER_PROVISIONAL,
    TIER_RESEARCH_ONLY,
    TIER_VALIDATED,
)
from src.training.evaluate import DEFAULT_CALIBRATION_FOLD, DEFAULT_DATA_DIR
from src.utils.ecg_labels import ECG_FOUNDER_LABELS

DEFAULT_METRICS_DIR = Path("results/metrics")
DEFAULT_PREFIX = "ptbxl_calibration_fold9"
DEFAULT_OUTPUT_PATH = Path("configs/calibration/ptbxl_fold9_v1.json")


def main() -> None:
    """CLI entry point for building the calibration artefact."""
    parser = argparse.ArgumentParser(
        description="Fit per-class Platt scaling and thresholds on the PTB-XL validation fold"
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fold", type=int, default=DEFAULT_CALIBRATION_FOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--minimum-positives",
        type=int,
        default=MINIMUM_CALIBRATION_POSITIVES,
        help=f"Support floor for attempting a fit (default: {MINIMUM_CALIBRATION_POSITIVES})",
    )
    args = parser.parse_args()

    if args.fold == 10:
        raise SystemExit(
            "Refusing to fit on fold 10 — it is the held-out audit fold. Use fold 9."
        )

    y_true, y_prob, provenance = load_fold_predictions(
        metrics_dir=args.metrics_dir,
        prefix=args.prefix,
        ptbxl_database_path=args.data_dir / "ptbxl_database.csv",
        labels_path=args.data_dir / "ecgfounder_ptbxl_label.csv",
    )
    print(
        f"Loaded fold {args.fold}: {provenance['records_matched']} labelled records "
        f"({provenance['records_unmatched']} without official labels)"
    )

    artifact = fit_calibration(
        y_true=y_true,
        y_prob=y_prob,
        class_names=ECG_FOUNDER_LABELS,
        fold=args.fold,
        ground_truth_source=provenance["ground_truth_source"],
        minimum_positives=args.minimum_positives,
    )
    artifact.save(args.output)

    summary = artifact.summary()
    counts = summary["tier_counts"]
    print(f"\nSaved calibration artefact to {args.output}")
    print(f"Heads with state: {summary['classes_with_state']}")
    for tier in (TIER_VALIDATED, TIER_PROVISIONAL, TIER_RESEARCH_ONLY):
        print(f"  {tier:<14} {counts.get(tier, 0)}")

    print("\nValidated heads (label — positives, AUROC, F1@0.5 → F1@learned):")
    validated = [artifact.classes[label] for label in summary["validated_labels"]]
    for entry in sorted(validated, key=lambda item: item.fit_f1 or 0.0, reverse=True):
        print(
            f"  {entry.label[:44]:<44} n={entry.positives:<5} "
            f"AUROC={entry.fit_auroc:.3f}  "
            f"F1 {entry.fit_f1_at_default_threshold:.3f} → {entry.fit_f1:.3f} "
            f"@ thr={entry.threshold:.3f}"
        )


if __name__ == "__main__":
    main()
