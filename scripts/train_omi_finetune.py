#!/usr/bin/env python3
# Fine-tune ECGFounder for OMI: frozen-backbone head training, then late-stage unfreezing.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.calibration.fit import select_threshold
from src.omi.dataset import load_labels
from src.omi.evaluation import (
    operating_point_metrics,
    patient_bootstrap_ci,
    ranking_metrics,
    subgroup_report,
)
from src.omi.model import build_classifier
from src.omi.signal_cache import build_signal_cache, open_signal_cache
from src.omi.training import (
    SignalDataset,
    TrainingConfig,
    predict,
    train_omi_classifier,
)
from src.pipeline.diagnose import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_FOLDS = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "omi" / "signal_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "omi"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "models" / "omi"

BASELINE_F1 = 0.396


def main() -> None:
    """CLI entry point for OMI fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune ECGFounder for OMI detection")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--unfrozen-stages", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--tag", type=str, default="v1")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    table = load_labels(args.data_dir, args.folds)
    _, valid = build_signal_cache(args.data_dir, table, args.cache_dir)
    signals = open_signal_cache(args.cache_dir)
    labels = table.OMI.to_numpy()

    # Undecodable recordings are dropped from both splits, never zero-filled.
    fold = table.fold.to_numpy()
    train_rows = np.flatnonzero((fold != args.validation_fold) & valid)
    validation_rows = np.flatnonzero((fold == args.validation_fold) & valid)
    validation_table = table.iloc[validation_rows].reset_index(drop=True)

    print(
        f"Train {len(train_rows)} ({int(labels[train_rows].sum())} OMI) | "
        f"Validation {len(validation_rows)} ({int(labels[validation_rows].sum())} OMI)"
    )

    config = TrainingConfig(
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        unfrozen_stages=args.unfrozen_stages,
        batch_size=args.batch_size,
    )
    model = build_classifier(args.model, device)
    best_state, history = train_omi_classifier(
        model, signals, labels, train_rows, validation_rows, device, config
    )
    model.load_state_dict(best_state)
    model.to(device)

    # Threshold comes from the training folds, never the fold being reported.
    train_loader = DataLoader(
        SignalDataset(signals, labels, train_rows), batch_size=args.batch_size, shuffle=False
    )
    validation_loader = DataLoader(
        SignalDataset(signals, labels, validation_rows),
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_scores = predict(model, train_loader, device)
    validation_scores = predict(model, validation_loader, device)
    threshold, train_f1 = select_threshold(labels[train_rows], train_scores)
    print(f"\nThreshold from training folds: {threshold:.4f} (train F1 {train_f1:.4f})")

    y_validation = labels[validation_rows]
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

    print(f"\n=== Fold {args.validation_fold} — fine-tuned ===")
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
        f"\n  Go/no-go — F1 95% CI lower bound above {BASELINE_F1}: "
        f"{'YES' if beats_baseline else 'NO'}"
    )

    subgroups = subgroup_report(validation_table, y_validation, validation_scores, threshold)
    print("\nPre-registered subgroups:")
    print(f"  {'SUBGROUP':<20} {'n':>6} {'OMI':>5} {'AUROC':>7} {'AUPRC':>7} {'F1':>7}")
    for name, values in subgroups.items():
        if values.get("auroc") is None:
            print(f"  {name:<20} {values['records']:>6} {values['omi']:>5}   (tek sınıf)")
            continue
        print(
            f"  {name:<20} {values['records']:>6} {values['omi']:>5} "
            f"{values['auroc']:>7.4f} {values['auprc']:>7.4f} {values['f1']:>7.4f}"
        )

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.weights_dir / f"omi_finetuned_{args.tag}.pt"
    torch.save(best_state, weights_path)

    report = {
        "tag": args.tag,
        "validation_fold": args.validation_fold,
        "config": vars(config),
        "train_records": int(len(train_rows)),
        "validation_records": int(len(validation_rows)),
        "threshold": threshold,
        "threshold_source": "training folds",
        "ranking": ranking,
        "operating_point": operating,
        "confidence_intervals": confidence,
        "baseline_f1": BASELINE_F1,
        "beats_baseline_on_validation": bool(beats_baseline),
        "subgroups": subgroups,
        "history": [vars(record) for record in history.epochs],
        "best_stage": history.best_stage,
        "best_epoch": history.best_epoch,
        "weights": str(weights_path.relative_to(PROJECT_ROOT)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"finetune_{args.tag}_fold{args.validation_fold}.json"
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWeights → {weights_path}")
    print(f"Report  → {output_path}")


if __name__ == "__main__":
    main()
