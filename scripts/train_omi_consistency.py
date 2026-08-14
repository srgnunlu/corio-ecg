#!/usr/bin/env python3
# Phase B step 5: teach the OMI head that a photograph of an ECG is the same ECG.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.omi.consistency import (
    ConsistencyConfig,
    PairedSignalDataset,
    predict_logits,
    predict_scores,
    train_consistency,
)
from src.omi.dataset import load_raw_signal
from src.omi.evaluation import operating_point_metrics, ranking_metrics
from src.omi.model import build_classifier
from src.omi.threshold import select_threshold_crossfit
from src.pipeline.diagnose import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "omi-corpus" / "clean" / "manifest.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "omi" / "omi_finetuned_v2a.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "omi" / "consistency_v1.json"
DEFAULT_SAVE_TO = PROJECT_ROOT / "models" / "omi" / "omi_consistency_v1.pt"

DEFAULT_HIDDEN_DIM = 256
NATURAL_PREVALENCE = 0.064


def _load_pairs(
    manifest: pd.DataFrame, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (clean, digitised, label) for every row of a corpus manifest."""
    clean: list[np.ndarray] = []
    digitised: list[np.ndarray] = []
    labels: list[int] = []
    for row in manifest.itertuples(index=False):
        path = Path(row.signal_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        digitised.append(np.load(path).astype(np.float16))
        clean.append(load_raw_signal(data_dir, row.ecg_row_record).astype(np.float16))
        labels.append(int(row.omi))
    return np.stack(clean), np.stack(digitised), np.array(labels)


def _evaluate(
    scores: np.ndarray, labels: np.ndarray, patients: np.ndarray, inherited: float
) -> dict:
    """Ranking plus both the inherited and a re-selected operating point."""
    crossfit = select_threshold_crossfit(
        labels,
        scores,
        patients,
        negative_weight=(labels == 1).sum()
        / (labels == 0).sum()
        * (1 - NATURAL_PREVALENCE)
        / NATURAL_PREVALENCE,
    )
    return {
        **ranking_metrics(labels, scores),
        "at_inherited_threshold": operating_point_metrics(labels, scores, inherited),
        "crossfit": crossfit.to_dict(),
    }


def main() -> None:
    """CLI entry point for paired clean/digitised consistency training."""
    parser = argparse.ArgumentParser(
        description="Train the OMI head on clean/digitised pairs"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-to", type=Path, default=DEFAULT_SAVE_TO)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--inherited-threshold", type=float, default=0.6423)
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=1.0,
        help="0 turns this into plain digitised fine-tuning — the ablation arm",
    )
    parser.add_argument("--clean-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    manifest = pd.read_csv(args.corpus)
    validation_manifest = manifest[manifest.fold == args.validation_fold]
    training_manifest = manifest[manifest.fold != args.validation_fold]
    if training_manifest.empty or validation_manifest.empty:
        raise SystemExit(
            f"Corpus holds folds {sorted(manifest.fold.unique())}; need both "
            f"fold {args.validation_fold} and at least one other."
        )
    overlap = set(training_manifest.patient_id) & set(validation_manifest.patient_id)
    if overlap:
        raise SystemExit(f"{len(overlap)} patients appear in both splits — split is leaky")

    print(
        f"Training on {len(training_manifest)} records "
        f"({int(training_manifest.omi.sum())} OMI, folds "
        f"{sorted(training_manifest.fold.unique())}), validating on "
        f"{len(validation_manifest)} ({int(validation_manifest.omi.sum())} OMI)"
    )

    train_clean, train_digitised, train_labels = _load_pairs(training_manifest, args.data_dir)
    _, validation_digitised, validation_labels = _load_pairs(
        validation_manifest, args.data_dir
    )
    validation_patients = validation_manifest.patient_id.to_numpy()

    device = get_device()
    model = build_classifier(args.model, device, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)

    baseline_scores = predict_scores(model, validation_digitised, device)
    baseline = _evaluate(
        baseline_scores, validation_labels, validation_patients, args.inherited_threshold
    )
    print(
        f"\nBefore training (v2a on digitised): AUROC {baseline['auroc']:.4f} "
        f"AUPRC {baseline['auprc']:.4f} "
        f"sens@crossfit {baseline['crossfit']['sensitivity']:.4f}"
    )

    # Freeze the target before a single weight moves: the consistency term aims
    # at what the clean-signal model said, not at whatever the student drifts to.
    teacher_logits = predict_logits(model, train_clean, device)
    print(
        f"  Teacher logits on clean training signals: "
        f"median positive {np.median(teacher_logits[train_labels == 1]):+.3f}, "
        f"median negative {np.median(teacher_logits[train_labels == 0]):+.3f}"
    )

    config = ConsistencyConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        consistency_weight=args.consistency_weight,
        clean_weight=args.clean_weight,
    )
    positives = int(train_labels.sum())
    positive_weight = (len(train_labels) - positives) / max(positives, 1)
    history, best = train_consistency(
        model,
        PairedSignalDataset(train_clean, train_digitised, train_labels, teacher_logits),
        validation_digitised,
        validation_labels,
        device,
        config,
        positive_weight=positive_weight,
    )

    print(f"\n  {'EPOCH':>5} {'TOTAL':>8} {'DIGIT':>8} {'CLEAN':>8} {'CONS':>8} {'AUPRC':>8}")
    for record in history:
        print(
            f"  {record.epoch:>5} {record.total_loss:>8.4f} {record.digitised_loss:>8.4f} "
            f"{record.clean_loss:>8.4f} {record.consistency_loss:>8.4f} "
            f"{record.validation_auprc:>8.4f}"
        )

    trained_scores = predict_scores(model, validation_digitised, device)
    trained = _evaluate(
        trained_scores, validation_labels, validation_patients, args.inherited_threshold
    )

    print(f"\n=== Digitised validation (fold {args.validation_fold}) ===")
    print(f"  {'':<12} {'AUROC':>8} {'AUPRC':>8} {'SENS':>8} {'SPEC':>8}")
    for name, values in (("before", baseline), ("after", trained)):
        print(
            f"  {name:<12} {values['auroc']:>8.4f} {values['auprc']:>8.4f} "
            f"{values['crossfit']['sensitivity']:>8.4f} "
            f"{values['crossfit']['specificity']:>8.4f}"
        )
    print(
        f"\n  Delta: AUROC {trained['auroc'] - baseline['auroc']:+.4f}  "
        f"AUPRC {trained['auprc'] - baseline['auprc']:+.4f}  "
        f"SENS {trained['crossfit']['sensitivity'] - baseline['crossfit']['sensitivity']:+.4f}"
    )

    # Score separation is the quantity the pilot found collapsing, so track it
    # directly rather than inferring it from the headline metrics.
    separation = {
        "before": float(
            np.median(baseline_scores[validation_labels == 1])
            - np.median(baseline_scores[validation_labels == 0])
        ),
        "after": float(
            np.median(trained_scores[validation_labels == 1])
            - np.median(trained_scores[validation_labels == 0])
        ),
    }
    print(
        f"  Median score separation (positive - negative): "
        f"{separation['before']:.4f} → {separation['after']:.4f}"
    )

    args.save_to.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_to)

    report = {
        "corpus": str(args.corpus),
        "training_records": int(len(training_manifest)),
        "training_positives": positives,
        "training_folds": sorted(int(f) for f in training_manifest.fold.unique()),
        "validation_fold": args.validation_fold,
        "validation_records": int(len(validation_manifest)),
        "consistency_weight": args.consistency_weight,
        "clean_weight": args.clean_weight,
        "teacher_logit_median_positive": float(np.median(teacher_logits[train_labels == 1])),
        "teacher_logit_median_negative": float(np.median(teacher_logits[train_labels == 0])),
        "epochs_run": len(history),
        "best_epoch": best,
        "history": [vars(record) for record in history],
        "baseline": baseline,
        "trained": trained,
        "median_score_separation": separation,
        "weights": str(args.save_to),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}\nWeights → {args.save_to}")


if __name__ == "__main__":
    main()
