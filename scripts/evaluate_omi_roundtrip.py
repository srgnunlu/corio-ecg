#!/usr/bin/env python3
# Phase B pilot: how much OMI performance survives the paper round-trip?

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.omi.dataset import load_labels, load_raw_signal
from src.omi.evaluation import operating_point_metrics, ranking_metrics
from src.omi.model import build_classifier
from src.pipeline.diagnose import build_paper_column_signals
from src.pipeline.digitize import ECGDigitiser
from src.pipeline.diagnose import get_device
from src.utils.ecg_render import DifficultyLevel, render_ecg_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_FOLDS = PROJECT_ROOT / "configs" / "omi" / "omi_folds_v1.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "omi" / "omi_finetuned_v2a.pt"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "processed" / "omi-roundtrip"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "omi" / "roundtrip_pilot.json"

DEFAULT_THRESHOLD = 0.6423
DEFAULT_HIDDEN_DIM = 256


def _select_balanced_subset(
    table, fold: int, negatives_per_positive: float, seed: int
):
    """Take every OMI positive in the fold plus a matched negative sample.

    The natural 6.4% prevalence would leave too few positives to measure a
    sensitivity drop on a pilot-sized run. Prevalence is identical for the
    clean and digitised arms, so the comparison stays valid — but absolute
    PPV and AUPRC are not comparable to the full-fold numbers.
    """
    validation = table[table.fold == fold]
    positives = validation[validation.OMI == 1]
    negatives = validation[validation.OMI == 0]
    n_negatives = min(len(negatives), int(round(len(positives) * negatives_per_positive)))
    sampled_negatives = negatives.sample(n=n_negatives, random_state=seed)
    subset = (
        pd.concat([positives, sampled_negatives])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return subset


@torch.no_grad()
def _score(model, signal: np.ndarray, device: torch.device) -> float:
    """Return the OMI probability for one preprocessed signal."""
    batch = torch.tensor(signal[None], dtype=torch.float32, device=device)
    return float(torch.sigmoid(model(batch)).item())


@torch.no_grad()
def _score_segment_ensemble(
    model, signal: np.ndarray, device: torch.device, layout_name: str
) -> float | None:
    """Score each printed paper column separately and average.

    Paper prints each column in a different time window; feeding all 12 leads
    at once mixes them. The production diagnosis path already does this and
    gained macro AUROC 0.75 -> 0.87 on PTB-XL.

    Returns None when the layout is not one the column map covers.
    """
    try:
        columns = build_paper_column_signals(signal, layout_name)
    except ValueError:
        return None
    batch = torch.tensor(np.stack(columns), dtype=torch.float32, device=device)
    return float(torch.sigmoid(model(batch)).mean().item())


def main() -> None:
    """CLI entry point for the OMI paper round-trip pilot."""
    parser = argparse.ArgumentParser(
        description="Measure OMI performance loss across render -> digitise"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in DifficultyLevel],
        default=DifficultyLevel.CLEAN.value,
    )
    parser.add_argument("--negatives-per-positive", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    table = load_labels(args.data_dir, args.folds)
    subset = _select_balanced_subset(
        table, args.fold, args.negatives_per_positive, args.seed
    )
    if args.max_records:
        subset = subset.head(args.max_records)

    positives = int(subset.OMI.sum())
    print(
        f"Pilot subset: {len(subset)} records, {positives} OMI "
        f"({positives / len(subset):.1%} — balanced, not the natural rate)"
    )

    image_dir = args.work_dir / "images" / args.difficulty
    image_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = build_classifier(args.model, device, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device).eval()
    digitiser = ECGDigitiser()

    clean_scores: list[float] = []
    digitised_scores: list[float] = []
    ensemble_scores: list[float] = []
    labels: list[int] = []
    failures: list[str] = []
    layouts: dict[str, int] = {}
    ensemble_unavailable = 0

    for record in tqdm(list(subset.itertuples(index=False)), desc="round-trip"):
        stem = record.ecg_row_record.removesuffix(".dat")
        record_path = args.data_dir / "row_data" / stem
        image_path = image_dir / f"{stem}.png"

        try:
            clean_signal = load_raw_signal(args.data_dir, record.ecg_row_record)
            if not image_path.exists():
                render_ecg_image(
                    str(record_path),
                    image_path,
                    difficulty=DifficultyLevel(args.difficulty),
                    seed=args.seed,
                )
            digitised_signal = digitiser.digitize(str(image_path))
        except Exception as error:  # noqa: BLE001 — one bad record must not stop the pilot
            failures.append(f"{stem}: {type(error).__name__}: {error}")
            continue

        layout_name = getattr(digitiser.last_info, "layout_name", "unknown")
        layouts[layout_name] = layouts.get(layout_name, 0) + 1

        digitised_score = _score(model, digitised_signal, device)
        ensemble_score = _score_segment_ensemble(
            model, digitised_signal, device, layout_name
        )
        if ensemble_score is None:
            # Fall back to the whole-signal score so the arms stay aligned.
            ensemble_score = digitised_score
            ensemble_unavailable += 1

        clean_scores.append(_score(model, clean_signal, device))
        digitised_scores.append(digitised_score)
        ensemble_scores.append(ensemble_score)
        labels.append(int(record.OMI))

    if failures:
        print(f"\n[WARN] {len(failures)} records failed the round-trip:")
        for failure in failures[:10]:
            print(f"  {failure}")

    y_true = np.array(labels)
    clean = np.array(clean_scores)
    digitised = np.array(digitised_scores)
    ensemble = np.array(ensemble_scores)

    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise SystemExit(
            f"Only {len(y_true)} records survived the round-trip "
            f"({int(y_true.sum())} positive) — not enough to compare arms. "
            "Check the failures listed above."
        )

    arms = {}
    for name, scores in (
        ("clean", clean),
        ("digitised", digitised),
        ("digitised_segment", ensemble),
    ):
        arms[name] = {
            **ranking_metrics(y_true, scores),
            **operating_point_metrics(y_true, scores, args.threshold),
        }

    print(f"\n=== Round-trip, difficulty={args.difficulty}, n={len(y_true)} ===")
    print(f"  {'ARM':<18} {'AUROC':>7} {'AUPRC':>7} {'SENS':>7} {'SPEC':>7} {'F1':>7}")
    for name, values in arms.items():
        print(
            f"  {name:<18} {values['auroc']:>7.4f} {values['auprc']:>7.4f} "
            f"{values['sensitivity']:>7.4f} {values['specificity']:>7.4f} {values['f1']:>7.4f}"
        )
    print(
        f"\n  Delta (digitised - clean): AUROC {arms['digitised']['auroc'] - arms['clean']['auroc']:+.4f}"
        f"  AUPRC {arms['digitised']['auprc'] - arms['clean']['auprc']:+.4f}"
        f"  SENS {arms['digitised']['sensitivity'] - arms['clean']['sensitivity']:+.4f}"
    )

    # How well does the score itself survive, record by record? A model that
    # keeps its ranking but shifts its scale needs recalibration, not retraining.
    correlation = float(np.corrcoef(clean, digitised)[0, 1]) if len(clean) > 1 else 0.0
    ensemble_correlation = (
        float(np.corrcoef(clean, ensemble)[0, 1]) if len(clean) > 1 else 0.0
    )
    print(f"  Per-record score correlation (clean vs digitised): {correlation:.4f}")
    print(f"  Per-record score correlation (clean vs segment):   {ensemble_correlation:.4f}")
    print(f"  Detected layouts: {layouts}")
    if ensemble_unavailable:
        print(f"  Segment-ensemble unavailable for {ensemble_unavailable} records (fell back)")

    report = {
        "difficulty": args.difficulty,
        "fold": args.fold,
        "records_attempted": int(len(subset)),
        "records_scored": int(len(y_true)),
        "failures": failures,
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "prevalence_note": "balanced subset — not the natural 6.4% rate",
        "threshold": args.threshold,
        "arms": arms,
        "score_correlation": correlation,
        "score_correlation_segment": ensemble_correlation,
        "detected_layouts": layouts,
        "segment_ensemble_unavailable": ensemble_unavailable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
