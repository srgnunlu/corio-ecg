#!/usr/bin/env python3
# Phase B step 8: consistency-train ECGFounder's 150-label head on PTB-XL pairs.
# Same recipe as the OMI head, now on the head the web app actually serves.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.omi.consistency import (
    ConsistencyConfig,
    PairedSignalDataset,
    predict_logits,
    predict_scores,
    train_consistency,
)
from src.pipeline.diagnose import get_device
from src.training.consistency_general import (
    HEADLINE_LABELS,
    PtbxlPairs,
    build_general_classifier,
    load_ptbxl_pairs,
    macro_scorer,
    macro_separation,
    mask_unsupported,
    per_label_auroc,
    positive_weights,
    segment_ensemble_scores,
    supported_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ptb-xl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "ptbxl-corpus" / "clean" / "manifest.csv"
DEFAULT_LABELS = DEFAULT_DATA_DIR / "ecgfounder_ptbxl_label.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "general" / "consistency_v1.json"
DEFAULT_SAVE_TO = PROJECT_ROOT / "models" / "general" / "consistency_v1_head.pt"


def _evaluate(model, pairs: PtbxlPairs, support: np.ndarray, device: torch.device) -> dict:
    """Macro metrics on the three views the product cares about.

    tiled: the digitised signal as the corpus stores it (all columns expanded);
    segment_ensemble: the web app's per-column path; clean: the WFDB signal,
    which must not get worse.
    """
    scorer = macro_scorer(support)
    views = {
        "tiled": predict_scores(model, pairs.digitised, device),
        "segment_ensemble": segment_ensemble_scores(
            model, pairs.digitised, pairs.layout_names, device
        ),
        "clean": predict_scores(model, pairs.clean, device),
    }
    report: dict = {}
    for name, scores in views.items():
        auprc, auroc = scorer(pairs.targets, scores)
        report[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "separation": macro_separation(pairs.targets, scores, support),
            "per_label_auroc": per_label_auroc(pairs.targets, scores, support),
        }
    # The summariser reads these top-level keys; the tiled view is the one the
    # model is trained on, so it is the headline.
    report["auroc"] = report["tiled"]["auroc"]
    report["auprc"] = report["tiled"]["auprc"]
    return report


def _print_views(label: str, report: dict) -> None:
    cells = []
    for view in ("tiled", "segment_ensemble", "clean"):
        values = report[view]
        cells.append(
            f"{view}: AUROC {values['auroc']:.4f} AUPRC {values['auprc']:.4f} "
            f"sep {values['separation']:.3f}"
        )
    print(f"  {label:<8} " + " | ".join(cells))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Consistency-train ECGFounder's 150-label head on PTB-XL pairs"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-to", type=Path, default=DEFAULT_SAVE_TO)
    parser.add_argument(
        "--validation-folds", type=int, nargs="+", default=[9, 10],
        help="PTB-XL strat_fold values held out; 9 and 10 match the calibration audit",
    )
    parser.add_argument("--min-train-positives", type=int, default=20)
    parser.add_argument("--min-validation-positives", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="a tenth of the OMI rate: this head starts pretrained, not from scratch",
    )
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    pairs = load_ptbxl_pairs(args.manifest, args.data_dir, args.labels, PROJECT_ROOT)
    validation_mask = np.isin(pairs.strat_fold, args.validation_folds)
    train = pairs.subset(~validation_mask)
    validation = pairs.subset(validation_mask)
    train_support = supported_labels(train.targets, args.min_train_positives)
    validation_support = supported_labels(validation.targets, args.min_validation_positives)
    # Scored labels must be trainable too, or the comparison mixes learned and
    # untouched outputs.
    scored = train_support & validation_support
    print(
        f"Training on {len(train)} records (folds {sorted(set(train.strat_fold))}), "
        f"validating on {len(validation)} (folds {args.validation_folds}); "
        f"{int(train_support.sum())} trainable labels, {int(scored.sum())} scored"
    )

    device = get_device()
    model = build_general_classifier(args.model, device)

    baseline = _evaluate(model, validation, scored, device)
    print("\nBefore training (ECGFounder as shipped):")
    _print_views("before", baseline)

    # Frozen targets from the untouched model on the clean recordings.
    teacher_logits = predict_logits(model, train.clean, device)

    config = ConsistencyConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        consistency_weight=args.consistency_weight,
        clean_weight=args.clean_weight,
    )
    history, best = train_consistency(
        model,
        PairedSignalDataset(
            train.clean,
            train.digitised,
            mask_unsupported(train.targets, train_support),
            teacher_logits,
        ),
        validation.digitised,
        mask_unsupported(validation.targets, scored),
        device,
        config,
        positive_weight=positive_weights(train.targets, train_support),
        validation_scorer=macro_scorer(scored),
    )
    print(
        f"\n  {'EPOCH':>5} {'TOTAL':>8} {'DIGIT':>8} {'CLEAN':>8} {'CONS':>8} "
        f"{'mAUPRC':>8} {'mAUROC':>8}"
    )
    for record in history:
        print(
            f"  {record.epoch:>5} {record.total_loss:>8.4f} {record.digitised_loss:>8.4f} "
            f"{record.clean_loss:>8.4f} {record.consistency_loss:>8.4f} "
            f"{record.validation_auprc:>8.4f} {record.validation_auroc:>8.4f}"
        )

    trained = _evaluate(model, validation, scored, device)
    print("\nAfter training:")
    _print_views("after", trained)
    print("\n  Headline labels, AUROC before -> after (tiled | segment-ensemble | clean):")
    for name in HEADLINE_LABELS:
        if name not in baseline["tiled"]["per_label_auroc"]:
            continue
        cells = " | ".join(
            f"{baseline[view]['per_label_auroc'][name]:.3f} -> "
            f"{trained[view]['per_label_auroc'][name]:.3f}"
            for view in ("tiled", "segment_ensemble", "clean")
        )
        print(f"    {name:<34} {cells}")

    args.save_to.parent.mkdir(parents=True, exist_ok=True)
    # Only the projection changed; the backbone is the shipped checkpoint.
    torch.save(model.backbone.dense.state_dict(), args.save_to)

    report = {
        "manifest": str(args.manifest),
        "training_records": int(len(train)),
        "validation_records": int(len(validation)),
        "validation_folds": args.validation_folds,
        "trainable_labels": int(train_support.sum()),
        "scored_labels": int(scored.sum()),
        "consistency_weight": args.consistency_weight,
        "clean_weight": args.clean_weight,
        "learning_rate": args.learning_rate,
        "epochs_run": len(history),
        "best_epoch": best,
        "history": [vars(record) for record in history],
        "baseline": baseline,
        "trained": trained,
        "median_score_separation": {
            "before": baseline["tiled"]["separation"],
            "after": trained["tiled"]["separation"],
        },
        "head_weights": str(args.save_to),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}\nHead weights → {args.save_to}")


if __name__ == "__main__":
    main()
