#!/usr/bin/env python3
# Produce the OMI submission CSV for the dataset's hidden test set.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.omi.dataset import load_labels
from src.omi.model import build_classifier
from src.omi.signal_cache import build_signal_cache, open_signal_cache
from src.omi.training import SignalDataset, predict
from src.pipeline.diagnose import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "ecgfounder" / "base" / "12_lead_ECGFounder.pth"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "omi" / "omi_finetuned_v2a.pt"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "omi" / "signal_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "omi" / "submission"

# From the winning variant's report: F2-max, fitted on the training folds only.
DEFAULT_THRESHOLD = 0.6423
DEFAULT_HIDDEN_DIM = 256


def main() -> None:
    """CLI entry point for building the test-set submission."""
    parser = argparse.ArgumentParser(
        description="Score the hidden OMI test set and write a submission CSV"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tag", type=str, default="v2a")
    args = parser.parse_args()

    device = get_device()
    test_table = load_labels(args.data_dir, table="test")
    print(f"Test records: {len(test_table)}")
    if "OMI" in test_table.columns:
        raise SystemExit("test.csv unexpectedly carries labels — check the dataset version")

    # The cache builder indexes by ecg_row_record, same as for the training set.
    _, valid = build_signal_cache(
        args.data_dir, test_table, args.cache_dir, name="test"
    )
    signals = open_signal_cache(args.cache_dir, name="test")
    if not valid.all():
        print(f"[WARN] {int((~valid).sum())} test recordings failed to decode")

    model = build_classifier(args.model, device, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)

    rows = np.arange(len(test_table))
    placeholder_labels = np.zeros(len(test_table))
    loader = DataLoader(
        SignalDataset(signals, placeholder_labels, rows),
        batch_size=args.batch_size,
        shuffle=False,
    )
    scores = predict(model, loader, device)

    # A recording we could not decode gets a negative call rather than a guess.
    predictions = (scores >= args.threshold).astype(int)
    predictions[~valid] = 0

    submission = pd.DataFrame(
        {
            "ecg_row_record": test_table.ecg_row_record.to_numpy(),
            "ecg_med_record": test_table.ecg_med_record.to_numpy(),
            "OMI": predictions,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / f"omi_submission_{args.tag}.csv"
    submission.to_csv(submission_path, index=False)

    positive_rate = float(predictions.mean())
    print(
        f"\nThreshold {args.threshold} → {int(predictions.sum())} positives "
        f"({positive_rate:.2%} of {len(predictions)})"
    )
    print(f"Score distribution: min {scores.min():.4f}, median {np.median(scores):.4f}, "
          f"max {scores.max():.4f}")
    print(f"\nSubmission → {submission_path}")

    metadata = {
        "tag": args.tag,
        "weights": str(args.weights.relative_to(PROJECT_ROOT)),
        "threshold": args.threshold,
        "threshold_source": "F2-max on training folds (fold 1-4)",
        "hidden_dim": args.hidden_dim,
        "records": int(len(test_table)),
        "predicted_positives": int(predictions.sum()),
        "predicted_positive_rate": positive_rate,
        "undecodable_records": int((~valid).sum()),
        "submission_file": submission_path.name,
    }
    with open(args.output_dir / f"omi_submission_{args.tag}_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
