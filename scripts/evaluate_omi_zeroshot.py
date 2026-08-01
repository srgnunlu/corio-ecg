#!/usr/bin/env python3
# Zero-shot reference: how well do stock ECGFounder heads separate OMI, before any training?

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.omi.dataset import load_labels, load_raw_signal
from src.pipeline.diagnose import ECGDiagnoser
from src.utils.ecg_labels import ECG_FOUNDER_LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_FOLDS = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "omi" / "zeroshot_fold0.json"

# Heads that could plausibly carry ischaemia/infarction signal, kept explicit so
# the reference is a stated hypothesis rather than a post-hoc search over 150.
CANDIDATE_HEADS: list[str] = [
    "ACUTE MI / STEMI",
    "ACUTE MI",
    "ST ELEVATION NOW PRESENT IN",
    "ANTERIOR INFARCT",
    "ANTEROSEPTAL INFARCT",
    "ANTEROLATERAL INFARCT",
    "INFERIOR INFARCT",
    "LATERAL INFARCT",
    "SEPTAL INFARCT",
    "POSTERIOR INFARCT",
    "INFERIOR-POSTERIOR INFARCT",
    "ANTERIOR INJURY PATTERN",
    "INFERIOR INJURY PATTERN",
    "LATERAL INJURY PATTERN",
    "INFEROLATERAL INJURY PATTERN",
    "ANTEROLATERAL INJURY PATTERN",
]


def _score(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC and average precision for one score vector."""
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
    }


def main() -> None:
    """CLI entry point for the zero-shot OMI reference."""
    parser = argparse.ArgumentParser(
        description="Measure stock ECGFounder heads against OMI on a validation fold"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Raw model outputs make the cleanest reference; the PTB-XL calibration was
    # fitted on a different population and would only confuse the comparison.
    os.environ["CORIO_CALIBRATION"] = "0"

    table = load_labels(args.data_dir, args.folds)
    validation = table[table.fold == args.fold].reset_index(drop=True)
    if args.max_records:
        validation = validation.head(args.max_records)

    positives = int(validation.OMI.sum())
    print(
        f"Fold {args.fold}: {len(validation)} records, {positives} OMI "
        f"({positives / len(validation):.2%} prevalence)"
    )

    diagnoser = ECGDiagnoser(checkpoint_path=args.model)
    probabilities = np.zeros((len(validation), len(ECG_FOUNDER_LABELS)), dtype=np.float32)

    for row_index, record in tqdm(
        enumerate(validation.itertuples(index=False)), total=len(validation)
    ):
        signal = load_raw_signal(args.data_dir, record.ecg_row_record)
        for result in diagnoser.diagnose_all(signal, apply_rate_adjustments=False):
            probabilities[row_index, result.index] = result.probability

    y_true = validation.OMI.to_numpy()
    label_index = {label: index for index, label in enumerate(ECG_FOUNDER_LABELS)}

    per_head = {}
    for head in CANDIDATE_HEADS:
        column = label_index[head]
        per_head[head] = _score(y_true, probabilities[:, column])

    candidate_columns = [label_index[head] for head in CANDIDATE_HEADS]
    combinations = {
        "max_of_candidates": _score(y_true, probabilities[:, candidate_columns].max(axis=1)),
        "mean_of_candidates": _score(y_true, probabilities[:, candidate_columns].mean(axis=1)),
    }

    ranked = sorted(per_head.items(), key=lambda item: -item[1]["auprc"])
    prevalence = float(y_true.mean())

    print(f"\nPrevalence (AUPRC floor): {prevalence:.4f}\n")
    print(f"{'HEAD':<44} {'AUROC':>7} {'AUPRC':>7}")
    for head, scores in ranked:
        print(f"{head[:44]:<44} {scores['auroc']:>7.4f} {scores['auprc']:>7.4f}")
    print()
    for name, scores in combinations.items():
        print(f"{name:<44} {scores['auroc']:>7.4f} {scores['auprc']:>7.4f}")

    # Does the model do better on the "obvious" STEMI-OMI than the hidden
    # NSTEMI-OMI? That gap is the clinical value proposition.
    best_head = ranked[0][0]
    best_scores = probabilities[:, label_index[best_head]]
    subgroups = {}
    for name, mask in {
        "stemi_labelled": validation.STEMI == 1,
        "nstemi_labelled": validation.NSTEMI == 1,
        "neither_stemi_nor_nstemi": (validation.STEMI == 0) & (validation.NSTEMI == 0),
    }.items():
        subset_true = y_true[mask.to_numpy()]
        if subset_true.sum() == 0 or subset_true.sum() == len(subset_true):
            subgroups[name] = {"records": int(mask.sum()), "omi": int(subset_true.sum())}
            continue
        subgroups[name] = {
            "records": int(mask.sum()),
            "omi": int(subset_true.sum()),
            **_score(subset_true, best_scores[mask.to_numpy()]),
        }

    print(f"\nSubgroups, scored with '{best_head}':")
    for name, values in subgroups.items():
        auroc = f"{values['auroc']:.4f}" if "auroc" in values else "n/a"
        auprc = f"{values['auprc']:.4f}" if "auprc" in values else "n/a"
        print(
            f"  {name:<28} n={values['records']:<6} OMI={values['omi']:<5} "
            f"AUROC={auroc} AUPRC={auprc}"
        )

    report = {
        "fold": args.fold,
        "records": int(len(validation)),
        "omi_positives": positives,
        "prevalence": prevalence,
        "calibration": "disabled (raw model outputs)",
        "per_head": per_head,
        "combinations": combinations,
        "best_head": best_head,
        "subgroups": subgroups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
