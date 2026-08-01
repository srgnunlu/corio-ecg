#!/usr/bin/env python3
# Compare threshold strategies on an already-trained OMI model. No retraining.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.omi.dataset import load_labels
from src.omi.evaluation import (
    operating_point_metrics,
    patient_bootstrap_ci,
    ranking_metrics,
    subgroup_report,
)
from src.omi.model import build_classifier
from src.omi.signal_cache import build_signal_cache, open_signal_cache
from src.omi.threshold import (
    BASELINE_SENSITIVITY,
    BASELINE_SPECIFICITY,
    select_threshold_at_min_sensitivity,
    select_threshold_at_min_specificity,
    select_threshold_fbeta,
)
from src.omi.training import SignalDataset, predict
from src.pipeline.diagnose import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_FOLDS = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "omi" / "omi_finetuned_v1.pt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "omi" / "signal_cache"
DEFAULT_SCORE_DIR = PROJECT_ROOT / "results" / "omi" / "scores"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "omi" / "threshold_strategies_fold0.json"

BASELINE = {
    "sensitivity": BASELINE_SENSITIVITY,
    "specificity": BASELINE_SPECIFICITY,
    "ppv": 0.277,
    "npv": 0.976,
    "f1": 0.396,
}


def _load_or_compute_scores(
    score_path: Path, model, signals, labels, rows, device, batch_size: int
) -> np.ndarray:
    """Score a row subset once and cache it — the model is fixed."""
    if score_path.exists():
        return np.load(score_path)
    loader = DataLoader(
        SignalDataset(signals, labels, rows), batch_size=batch_size, shuffle=False
    )
    scores = predict(model, loader, device)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_path, scores)
    return scores


def main() -> None:
    """CLI entry point for threshold-strategy comparison."""
    parser = argparse.ArgumentParser(
        description="Compare OMI threshold strategies without retraining"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--tag", type=str, default="v1")
    args = parser.parse_args()

    device = get_device()
    table = load_labels(args.data_dir, args.folds)
    _, valid = build_signal_cache(args.data_dir, table, args.cache_dir)
    signals = open_signal_cache(args.cache_dir)
    labels = table.OMI.to_numpy()

    fold = table.fold.to_numpy()
    train_rows = np.flatnonzero((fold != args.validation_fold) & valid)
    validation_rows = np.flatnonzero((fold == args.validation_fold) & valid)
    validation_table = table.iloc[validation_rows].reset_index(drop=True)

    model = build_classifier(args.model, device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)

    train_scores = _load_or_compute_scores(
        args.score_dir / f"train_{args.tag}_fold{args.validation_fold}.npy",
        model, signals, labels, train_rows, device, args.batch_size,
    )
    validation_scores = _load_or_compute_scores(
        args.score_dir / f"validation_{args.tag}_fold{args.validation_fold}.npy",
        model, signals, labels, validation_rows, device, args.batch_size,
    )
    y_train = labels[train_rows]
    y_validation = labels[validation_rows]
    patients = validation_table.Patient_id.to_numpy()

    ranking = ranking_metrics(y_validation, validation_scores)
    print(
        f"Model (unchanged): AUROC {ranking['auroc']:.4f}  AUPRC {ranking['auprc']:.4f}\n"
        "Thresholds are selected on the training folds only.\n"
    )

    # Every strategy is fitted on the training folds, then reported on fold 0.
    strategies = [
        select_threshold_fbeta(y_train, train_scores, beta=1.0),
        select_threshold_fbeta(y_train, train_scores, beta=2.0),
        select_threshold_at_min_sensitivity(y_train, train_scores, BASELINE_SENSITIVITY),
        select_threshold_at_min_specificity(y_train, train_scores, BASELINE_SPECIFICITY),
    ]

    print(
        f"  {'STRATEGY':<38} {'thr':>6} {'SENS':>7} {'SPEC':>7} {'PPV':>7} {'NPV':>7} {'F1':>7}"
    )
    print(
        f"  {'[published baseline]':<38} {'—':>6} "
        f"{BASELINE['sensitivity']:>7.4f} {BASELINE['specificity']:>7.4f} "
        f"{BASELINE['ppv']:>7.4f} {BASELINE['npv']:>7.4f} {BASELINE['f1']:>7.4f}"
    )

    results = []
    for choice in strategies:
        metrics = operating_point_metrics(y_validation, validation_scores, choice.threshold)
        confidence = {
            name: patient_bootstrap_ci(
                y_validation, validation_scores, patients,
                lambda t, s, key=name: operating_point_metrics(t, s, choice.threshold)[key],
                rounds=args.bootstrap_rounds,
            )
            for name in ("sensitivity", "specificity", "f1")
        }
        print(
            f"  {choice.strategy:<38} {choice.threshold:>6.3f} "
            f"{metrics['sensitivity']:>7.4f} {metrics['specificity']:>7.4f} "
            f"{metrics['ppv']:>7.4f} {metrics['npv']:>7.4f} {metrics['f1']:>7.4f}"
        )
        results.append(
            {
                "selection": choice.to_dict(),
                "validation": metrics,
                "confidence_intervals": confidence,
                "subgroups": subgroup_report(
                    validation_table, y_validation, validation_scores, choice.threshold
                ),
            }
        )

    # The clinically decisive comparison: at the baseline's detection rate, is
    # the false-alarm burden lower?
    iso = next(r for r in results if r["selection"]["strategy"].startswith("max_specificity"))
    iso_metrics = iso["validation"]
    iso_ci = iso["confidence_intervals"]
    print(
        f"\nIso-sensitivity comparison (target sens {BASELINE_SENSITIVITY}):\n"
        f"  achieved sens {iso_metrics['sensitivity']:.4f} "
        f"[{iso_ci['sensitivity']['lower']:.4f}, {iso_ci['sensitivity']['upper']:.4f}]\n"
        f"  spec {iso_metrics['specificity']:.4f} "
        f"[{iso_ci['specificity']['lower']:.4f}, {iso_ci['specificity']['upper']:.4f}] "
        f"vs baseline {BASELINE['specificity']}"
    )
    beats_on_specificity = iso_ci["specificity"]["lower"] > BASELINE["specificity"]
    print(
        f"  Specificity 95% CI lower bound above baseline: "
        f"{'YES' if beats_on_specificity else 'NO'}"
    )

    report = {
        "tag": args.tag,
        "validation_fold": args.validation_fold,
        "model_ranking": ranking,
        "baseline": BASELINE,
        "strategies": results,
        "iso_sensitivity_beats_baseline_specificity": bool(beats_on_specificity),
        "note": (
            "Thresholds fitted on training folds, reported on the validation fold. "
            "The official test set was not touched."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
