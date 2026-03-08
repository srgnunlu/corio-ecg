# Round-trip evaluation — compares ECGFounder diagnosis on clean vs digitized signals
# Measures how much diagnostic accuracy degrades through the image→digitize pipeline

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from tqdm import tqdm

from src.pipeline.diagnose import ECGDiagnoser
from src.utils.ecg_labels import NUM_CLASSES
from src.utils.metrics import compute_pearson_per_lead
from src.utils.wfdb_helpers import read_ecg_signal

DEFAULT_MODEL_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")
DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_DIGITIZED_DIR = Path("data/processed/signals")
DEFAULT_THRESHOLD = 0.5

# Roundtrip scenario names and their subdirectories
ROUNDTRIP_SCENARIOS: dict[str, str] = {
    "clean_roundtrip": "clean",
    "moderate_roundtrip": "moderate",
    "hard_roundtrip": "hard",
}


def _extract_probabilities(
    diagnoser: ECGDiagnoser, signal: np.ndarray
) -> np.ndarray:
    """Run ECGFounder on a signal and return the full 150-class probability vector.

    Args:
        diagnoser: Loaded ECGDiagnoser instance.
        signal: Z-score normalized array with shape (12, 5000).

    Returns:
        Probability array with shape (150,).
    """
    results = diagnoser.diagnose_all(signal)
    probabilities = np.zeros(NUM_CLASSES)
    for result in results:
        probabilities[result.index] = result.probability
    return probabilities


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, handling zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(1.0 - cosine_distance(a, b))


def _agreement_rate(
    baseline_probs: np.ndarray,
    roundtrip_probs: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of classes where baseline and roundtrip agree (both above or below threshold)."""
    baseline_positive = baseline_probs >= threshold
    roundtrip_positive = roundtrip_probs >= threshold
    return float(np.mean(baseline_positive == roundtrip_positive))


def load_test_fold_ids(data_dir: Path, max_samples: int | None) -> list[tuple[int, str]]:
    """Load PTB-XL test fold ecg_ids and their file paths.

    Args:
        data_dir: Path to PTB-XL dataset root.
        max_samples: Optional limit on number of records.

    Returns:
        List of (ecg_id, record_path) tuples for test fold records.
    """
    import ast
    import pandas as pd

    csv_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(csv_path, index_col="ecg_id")
    test_df = df[df.strat_fold == 10]

    if max_samples is not None:
        test_df = test_df.head(max_samples)

    records: list[tuple[int, str]] = []
    for ecg_id, row in test_df.iterrows():
        record_path = str(data_dir / row.filename_hr)
        if record_path.endswith(".hea"):
            record_path = record_path[:-4]
        records.append((int(ecg_id), record_path))

    return records


def evaluate_roundtrip(
    data_dir: Path,
    digitized_dir: Path,
    model_path: Path,
    threshold: float,
    max_samples: int | None = None,
) -> dict:
    """Run round-trip evaluation comparing clean vs digitized diagnoses.

    Args:
        data_dir: Path to PTB-XL dataset root.
        digitized_dir: Path to digitized signals directory.
        model_path: Path to ECGFounder checkpoint.
        threshold: Probability threshold for agreement rate calculation.
        max_samples: Optional limit on records to process.

    Returns:
        Results dictionary with per-scenario comparison metrics.
    """
    records = load_test_fold_ids(data_dir, max_samples)
    print(f"Evaluating {len(records)} test records...")

    # Load model once for all scenarios
    diagnoser = ECGDiagnoser(checkpoint_path=model_path)

    # Run baseline inference on clean WFDB signals
    baseline_probs: dict[int, np.ndarray] = {}
    baseline_signals: dict[int, np.ndarray] = {}
    baseline_failures = 0

    print("\n--- Baseline (clean WFDB signals) ---")
    for ecg_id, record_path in tqdm(records, desc="Baseline"):
        try:
            signal, _ = read_ecg_signal(record_path)
            baseline_probs[ecg_id] = _extract_probabilities(diagnoser, signal)
            baseline_signals[ecg_id] = signal
        except Exception as error:
            print(f"  Warning: baseline failed for ecg_id={ecg_id}: {error}")
            baseline_failures += 1

    # Build baseline summary
    baseline_prob_values = list(baseline_probs.values())
    results: dict = {
        "n_records": len(records),
        "threshold": threshold,
        "scenarios": {
            "baseline": {
                "description": "Clean PTB-XL WFDB signals",
                "n_successful": len(baseline_probs),
                "mean_probability": float(np.mean(np.stack(baseline_prob_values)))
                if baseline_prob_values else 0.0,
            },
        },
    }

    # Run each roundtrip scenario
    for scenario_name, subdir in ROUNDTRIP_SCENARIOS.items():
        scenario_dir = digitized_dir / subdir
        if not scenario_dir.exists():
            print(f"\n--- {scenario_name}: directory {scenario_dir} not found, skipping ---")
            continue

        print(f"\n--- {scenario_name} ({subdir}) ---")
        cosine_similarities: list[float] = []
        abs_prob_diffs: list[float] = []
        agreement_rates: list[float] = []
        pearson_correlations: list[float] = []
        n_successful = 0

        for ecg_id, _ in tqdm(records, desc=scenario_name):
            # Skip records that failed baseline
            if ecg_id not in baseline_probs:
                continue

            npy_path = scenario_dir / f"{ecg_id}.npy"
            if not npy_path.exists():
                continue

            try:
                digitized_signal = np.load(npy_path)

                # Compute signal quality: Pearson correlation between z-scored signals
                # Both clean and digitized are z-score normalized
                per_lead_r = compute_pearson_per_lead(
                    baseline_signals[ecg_id], digitized_signal
                )
                pearson_correlations.append(float(np.nanmean(per_lead_r)))

                # Run diagnosis on the digitized signal
                roundtrip_probs = _extract_probabilities(diagnoser, digitized_signal)

                # Compare probability vectors
                cos_sim = _cosine_similarity(baseline_probs[ecg_id], roundtrip_probs)
                cosine_similarities.append(cos_sim)

                abs_diff = float(np.mean(np.abs(baseline_probs[ecg_id] - roundtrip_probs)))
                abs_prob_diffs.append(abs_diff)

                agree = _agreement_rate(baseline_probs[ecg_id], roundtrip_probs, threshold)
                agreement_rates.append(agree)

                n_successful += 1

            except Exception as error:
                print(f"  Warning: {scenario_name} failed for ecg_id={ecg_id}: {error}")

        # Build scenario summary
        scenario_result: dict = {
            "description": f"Synthetic image ({subdir} difficulty) -> digitized",
            "n_successful": n_successful,
        }
        if n_successful > 0:
            scenario_result["mean_cosine_similarity_vs_baseline"] = round(
                float(np.mean(cosine_similarities)), 4
            )
            scenario_result["mean_abs_prob_diff_vs_baseline"] = round(
                float(np.mean(abs_prob_diffs)), 6
            )
            scenario_result["agreement_rate_vs_baseline"] = round(
                float(np.mean(agreement_rates)), 4
            )
            scenario_result["mean_pearson_correlation"] = round(
                float(np.nanmean(pearson_correlations)), 4
            )

        results["scenarios"][scenario_name] = scenario_result

    # TODO: Add per-class breakdown of probability differences
    # TODO: Compute signal quality using raw (pre-normalized) signals for proper SNR

    return results


def _print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 75)
    print("ROUND-TRIP EVALUATION RESULTS")
    print("=" * 75)
    print(f"Records: {results['n_records']}  |  Threshold: {results['threshold']}")
    print("-" * 75)

    header = f"{'Scenario':<22} {'N':>5} {'CosSim':>8} {'AbsDiff':>9} {'Agree%':>8} {'Pearson':>8}"
    print(header)
    print("-" * 75)

    for name, data in results["scenarios"].items():
        n = data["n_successful"]
        cos_sim = data.get("mean_cosine_similarity_vs_baseline", "—")
        abs_diff = data.get("mean_abs_prob_diff_vs_baseline", "—")
        agree = data.get("agreement_rate_vs_baseline", "—")
        pearson = data.get("mean_pearson_correlation", "—")

        # Format numeric values
        cos_str = f"{cos_sim:.4f}" if isinstance(cos_sim, float) else cos_sim
        diff_str = f"{abs_diff:.6f}" if isinstance(abs_diff, float) else abs_diff
        agree_str = f"{agree:.4f}" if isinstance(agree, float) else agree
        pear_str = f"{pearson:.4f}" if isinstance(pearson, float) else pearson

        print(f"{name:<22} {n:>5} {cos_str:>8} {diff_str:>9} {agree_str:>8} {pear_str:>8}")

    print("=" * 75)


def main() -> None:
    """CLI entry point for round-trip evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare ECGFounder diagnosis on clean vs digitized signals"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Path to PTB-XL dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--digitized-dir", type=Path, default=DEFAULT_DIGITIZED_DIR,
        help=f"Path to digitized signals (default: {DEFAULT_DIGITIZED_DIR})",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Path to ECGFounder checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Probability threshold for agreement rate (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of test records (default: all)",
    )
    args = parser.parse_args()

    results = evaluate_roundtrip(
        data_dir=args.data_dir,
        digitized_dir=args.digitized_dir,
        model_path=args.model,
        threshold=args.threshold,
        max_samples=args.max_samples,
    )

    # Save results JSON
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roundtrip_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    _print_comparison_table(results)


if __name__ == "__main__":
    main()
