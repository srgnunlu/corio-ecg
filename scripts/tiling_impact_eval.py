"""Isolate the diagnostic cost of paper-ECG segment tiling.

ECGFounder was trained on genuine 10 s PTB-XL recordings where every lead has
the full 10 s of continuous beats. A paper ECG only prints ~2.5 s per lead (one
column of a 3x4 layout; ~5 s for some layouts), so the pipeline tiles that short
segment 2-4x to fake a 10 s input. This script measures how much that tiling
alone degrades diagnosis and heart-rate estimation — with NO digitization noise,
by tiling the clean PTB-XL signals directly.

Three arms scored against the same PTB-XL ground truth:
- ``genuine``  : the real 10 s signal (identical to the baseline eval).
- ``tile_2_5s``: first 2.5 s tiled 4x (simulates a 3x4 column lead).
- ``tile_5s``  : first 5 s tiled 2x (simulates a wider layout / 2-column print).

It also reports, per arm, the heart rate vs the genuine-signal heart rate, to
quantify the HR doubling/halving the tiling introduces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.pipeline.diagnose import ECGDiagnoser
from src.training.evaluate import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OFFICIAL_LABELS_PATH,
    build_ground_truth_evaluation,
    build_official_ground_truth_evaluation,
    load_ptbxl_metadata,
)
from src.utils.ecg_labels import NUM_CLASSES
from src.utils.rhythm import estimate_heart_rate_bpm
from src.utils.wfdb_helpers import read_ecg_signal

DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_OUTPUT_DIR = Path("results/metrics")
TARGET_LEN = 5000

# Rhythm classes a clinician cares about first — reported individually because
# macro metrics over 35 classes hide what happens to the common diagnoses.
_KEY_CLASSES = (
    "NORMAL ECG",
    "SINUS RHYTHM",
    "SINUS BRADYCARDIA",
    "SINUS TACHYCARDIA",
    "ATRIAL FIBRILLATION",
    "PREMATURE VENTRICULAR COMPLEXES",
)


def tile_segment(signal: np.ndarray, segment_len: int, total: int = TARGET_LEN) -> np.ndarray:
    """Tile the first ``segment_len`` samples of each lead to fill ``total``."""
    segment = signal[:, :segment_len]
    repeats = int(np.ceil(total / segment_len))
    return np.tile(segment, (1, repeats))[:, :total]


def _classification_block(evaluation: dict) -> dict:
    """Pull the headline metrics + key-class F1 out of an evaluation dict."""
    classification = evaluation["classification"]
    per_class = classification.get("per_class", {})
    return {
        "macro_auroc": classification["macro_auroc"],
        "macro_average_precision": classification["macro_average_precision"],
        "micro_f1": classification["micro_f1"],
        "macro_f1": classification["macro_f1"],
        "key_class_f1": {
            name: per_class.get(name, {}).get("f1")
            for name in _KEY_CLASSES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metadata = load_ptbxl_metadata(args.data_dir)
    test_records = metadata[metadata.strat_fold == 10]
    if args.max_samples is not None:
        test_records = test_records.head(args.max_samples)
    print(f"Evaluating {len(test_records)} PTB-XL test records across 3 length arms...")

    diagnoser = ECGDiagnoser(checkpoint_path=args.model)

    arms = {
        "genuine": lambda s: s,
        "tile_2_5s": lambda s: tile_segment(s, 1250),
        "tile_5s": lambda s: tile_segment(s, 2500),
    }
    probabilities: dict[str, list[np.ndarray]] = {name: [] for name in arms}
    heart_rates: dict[str, list[float | None]] = {name: [] for name in arms}
    scp_codes: list[dict[str, float]] = []
    filenames: list[str] = []

    for ecg_id, row in tqdm(test_records.iterrows(), total=len(test_records)):
        record_path = str(args.data_dir / row.filename_hr)
        if record_path.endswith(".hea"):
            record_path = record_path[:-4]
        try:
            signal, _ = read_ecg_signal(record_path)
        except Exception as error:  # noqa: BLE001 - skip unreadable records
            print(f"Warning: failed ecg_id={ecg_id}: {error}")
            continue

        for name, transform in arms.items():
            arm_signal = transform(signal)
            results = diagnoser.diagnose_all(arm_signal, apply_rate_adjustments=False)
            probability = np.zeros(NUM_CLASSES, dtype=np.float32)
            for result in results:
                probability[result.index] = result.probability
            probabilities[name].append(probability)
            heart_rates[name].append(estimate_heart_rate_bpm(arm_signal))

        scp_codes.append(row.scp_codes)
        filenames.append(str(row.filename_hr))

    summary: dict = {"records_scored": len(scp_codes), "arms": {}}
    for name in arms:
        matrix = np.stack(probabilities[name], axis=0)
        semantic = build_ground_truth_evaluation(scp_codes, matrix, threshold=args.threshold)
        arm_summary = {"semantic_scp_subset": _classification_block(semantic)}
        if DEFAULT_OFFICIAL_LABELS_PATH.exists():
            official = build_official_ground_truth_evaluation(
                filenames, matrix, DEFAULT_OFFICIAL_LABELS_PATH, threshold=args.threshold
            )
            arm_summary["official_150_class"] = _classification_block(official)
        summary["arms"][name] = arm_summary

    # Heart-rate breakage vs the genuine signal's own estimate.
    genuine_hr = heart_rates["genuine"]
    for name in ("tile_2_5s", "tile_5s"):
        diffs = [
            abs(h - g)
            for h, g in zip(heart_rates[name], genuine_hr, strict=True)
            if h is not None and g is not None
        ]
        diffs_array = np.asarray(diffs)
        summary["arms"][name]["heart_rate_vs_genuine"] = {
            "n_compared": len(diffs),
            "within_5_bpm_frac": float((diffs_array <= 5).mean()) if len(diffs) else None,
            "over_20_bpm_frac": float((diffs_array > 20).mean()) if len(diffs) else None,
            "median_abs_diff_bpm": float(np.median(diffs_array)) if len(diffs) else None,
            "max_abs_diff_bpm": float(diffs_array.max()) if len(diffs) else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "tiling_impact_eval.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")

    print("\n=== Diagnosis (semantic SCP subset) ===")
    print(f"{'arm':<12}{'macroAUROC':>12}{'macroAP':>10}{'microF1':>10}")
    for name in arms:
        c = summary["arms"][name]["semantic_scp_subset"]
        print(f"{name:<12}{c['macro_auroc']:>12.3f}{c['macro_average_precision']:>10.3f}{c['micro_f1']:>10.3f}")
    print("\n=== Heart rate vs genuine ===")
    for name in ("tile_2_5s", "tile_5s"):
        h = summary["arms"][name]["heart_rate_vs_genuine"]
        print(
            f"{name:<12} within5={h['within_5_bpm_frac']:.0%}  "
            f">20bpm={h['over_20_bpm_frac']:.0%}  "
            f"median={h['median_abs_diff_bpm']:.1f}  max={h['max_abs_diff_bpm']:.0f}"
        )


if __name__ == "__main__":
    main()
