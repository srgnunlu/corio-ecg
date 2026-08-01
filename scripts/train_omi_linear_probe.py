#!/usr/bin/env python3
# Linear probe on frozen ECGFounder features: does the backbone already encode OMI?

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.calibration.fit import select_threshold
from src.omi.dataset import load_labels, select_fold
from src.omi.evaluation import (
    operating_point_metrics,
    patient_bootstrap_ci,
    ranking_metrics,
    subgroup_report,
)
from src.omi.features import INPUT_MODES, INPUT_RAW, cached_features, extract_features
from src.pipeline.diagnose import ECGDiagnoser, get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_FOLDS = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "omi" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "omi"

# Published baseline on the hidden test set — the bar Gate 2 has to clear.
BASELINE_F1 = 0.396


def main() -> None:
    """CLI entry point for the frozen-feature linear probe."""
    parser = argparse.ArgumentParser(
        description="Train a logistic regression on frozen ECGFounder features"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default=INPUT_RAW)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()

    table = load_labels(args.data_dir, args.folds)
    print(f"Loaded {len(table)} development records")

    device = get_device()
    diagnoser = ECGDiagnoser(checkpoint_path=args.model, device=device)
    backbone = diagnoser.model

    # One cache per input mode, covering the whole development set in table order.
    cache_path = args.cache_dir / f"features_{args.input_mode}.npy"
    features, valid = cached_features(
        cache_path,
        lambda: extract_features(
            backbone, device, args.data_dir, table, input_mode=args.input_mode
        ),
    )
    if len(features) != len(table):
        raise SystemExit(
            f"Cached features ({len(features)}) do not match the table ({len(table)}). "
            f"Delete {cache_path} and re-run."
        )
    print(f"Features: {features.shape} ({args.input_mode})")

    # Undecodable recordings carry a zero feature row; drop them everywhere so
    # they cannot silently act as a constant class.
    if not valid.all():
        dropped = int((~valid).sum())
        print(f"Excluding {dropped} recordings that failed to decode")
        table = table[valid].reset_index(drop=True)
        features = features[valid]

    validation_mask = (table.fold == args.validation_fold).to_numpy()
    train_features, validation_features = features[~validation_mask], features[validation_mask]
    train_table = select_fold(table, args.validation_fold, holdout=False)
    validation_table = select_fold(table, args.validation_fold, holdout=True)
    y_train = train_table.OMI.to_numpy()
    y_validation = validation_table.OMI.to_numpy()

    print(
        f"Train {len(y_train)} ({int(y_train.sum())} OMI) | "
        f"Validation {len(y_validation)} ({int(y_validation.sum())} OMI)"
    )

    scaler = StandardScaler().fit(train_features)
    # class_weight balances the 6.4% prevalence; without it the probe collapses
    # to predicting "no OMI" for everything.
    probe = LogisticRegression(max_iter=2000, class_weight="balanced")
    probe.fit(scaler.transform(train_features), y_train)

    train_scores = probe.predict_proba(scaler.transform(train_features))[:, 1]
    validation_scores = probe.predict_proba(scaler.transform(validation_features))[:, 1]

    # Threshold is selected on training folds only — never on the fold we report.
    threshold, train_f1 = select_threshold(y_train, train_scores)
    print(f"\nThreshold from training folds: {threshold:.4f} (train F1 {train_f1:.4f})")

    ranking = ranking_metrics(y_validation, validation_scores)
    operating = operating_point_metrics(y_validation, validation_scores, threshold)
    patients = validation_table.Patient_id.to_numpy()

    confidence = {
        "auroc": patient_bootstrap_ci(
            y_validation, validation_scores, patients,
            lambda t, s: float(roc_auc_score(t, s)), rounds=args.bootstrap_rounds,
        ),
        "auprc": patient_bootstrap_ci(
            y_validation, validation_scores, patients,
            lambda t, s: float(average_precision_score(t, s)), rounds=args.bootstrap_rounds,
        ),
        "f1": patient_bootstrap_ci(
            y_validation, validation_scores, patients,
            lambda t, s: operating_point_metrics(t, s, threshold)["f1"],
            rounds=args.bootstrap_rounds,
        ),
    }

    print(f"\n=== Fold {args.validation_fold} ({args.input_mode}) ===")
    print(
        f"  AUROC {ranking['auroc']:.4f} "
        f"[{confidence['auroc']['lower']:.4f}, {confidence['auroc']['upper']:.4f}]"
    )
    print(
        f"  AUPRC {ranking['auprc']:.4f} "
        f"[{confidence['auprc']['lower']:.4f}, {confidence['auprc']['upper']:.4f}]"
    )
    print(
        f"  F1    {operating['f1']:.4f} "
        f"[{confidence['f1']['lower']:.4f}, {confidence['f1']['upper']:.4f}]"
        f"   (baseline {BASELINE_F1})"
    )
    print(
        f"  Sens {operating['sensitivity']:.4f}  Spec {operating['specificity']:.4f}  "
        f"PPV {operating['ppv']:.4f}  NPV {operating['npv']:.4f}"
    )

    beats_baseline = confidence["f1"]["lower"] > BASELINE_F1
    print(
        f"\n  Go/no-go — F1 95% CI lower bound above baseline {BASELINE_F1}: "
        f"{'YES' if beats_baseline else 'NO'}"
    )
    print("  (Validation-fold result; the hidden test set is a different population.)")

    subgroups = subgroup_report(validation_table, y_validation, validation_scores, threshold)
    print("\nPre-registered subgroups:")
    print(f"  {'SUBGROUP':<20} {'n':>6} {'OMI':>5} {'AUROC':>7} {'AUPRC':>7} {'F1':>7}")
    for name, values in subgroups.items():
        if "auroc" not in values or values["auroc"] is None:
            print(f"  {name:<20} {values['records']:>6} {values['omi']:>5}   (tek sınıf)")
            continue
        print(
            f"  {name:<20} {values['records']:>6} {values['omi']:>5} "
            f"{values['auroc']:>7.4f} {values['auprc']:>7.4f} {values['f1']:>7.4f}"
        )

    report = {
        "input_mode": args.input_mode,
        "validation_fold": args.validation_fold,
        "train_records": int(len(y_train)),
        "validation_records": int(len(y_validation)),
        "threshold": threshold,
        "threshold_source": "training folds",
        "ranking": ranking,
        "operating_point": operating,
        "confidence_intervals": confidence,
        "baseline_f1": BASELINE_F1,
        "beats_baseline_on_validation": bool(beats_baseline),
        "subgroups": subgroups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"linear_probe_{args.input_mode}_fold{args.validation_fold}.json"
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
