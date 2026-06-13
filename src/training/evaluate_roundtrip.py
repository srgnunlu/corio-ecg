# Round-trip evaluation — compares ECGFounder diagnosis on clean vs digitized signals
# Measures how much diagnostic accuracy degrades through the image→digitize pipeline

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from tqdm import tqdm

from src.pipeline.diagnose import ECGDiagnoser
from src.pipeline.lead_assignment import LAYOUT_3X4, LAYOUT_6X2
from src.training.evaluate import (
    DEFAULT_OFFICIAL_LABELS_PATH,
    build_ground_truth_evaluation,
    build_official_ground_truth_evaluation,
    load_ptbxl_metadata,
)
from src.utils.ecg_labels import NUM_CLASSES
from src.utils.metrics import compute_pearson_per_lead
from src.utils.wfdb_helpers import read_ecg_signal

DEFAULT_MODEL_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")
DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_DIGITIZED_DIR = Path("data/processed/signals")
DEFAULT_THRESHOLD = 0.5
DEFAULT_PAPER_LAYOUT = "3x4"
DEFAULT_SEGMENT_AGGREGATION = "mean"

# Roundtrip scenario names and their subdirectories
ROUNDTRIP_SCENARIOS: dict[str, str] = {
    "clean_roundtrip": "clean",
    "moderate_roundtrip": "moderate",
    "hard_roundtrip": "hard",
}
PAPER_LAYOUTS: dict[str, list[list[int]]] = {
    "3x4": LAYOUT_3X4,
    "6x2": LAYOUT_6X2,
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
    results = diagnoser.diagnose_all(signal, apply_rate_adjustments=False)
    probabilities = np.zeros(NUM_CLASSES, dtype=np.float32)
    for result in results:
        probabilities[result.index] = result.probability
    return probabilities


def _build_layout_column_signals(
    signal: np.ndarray,
    layout_name: str,
) -> list[np.ndarray]:
    """Build one sparse 12-lead model input for each printed paper column."""
    matched_layout = next(
        (lead_rows for name, lead_rows in PAPER_LAYOUTS.items() if name in layout_name),
        None,
    )
    if matched_layout is None:
        raise ValueError(f"Unsupported paper layout: {layout_name}")
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError("signal must have shape (12, samples)")

    column_signals: list[np.ndarray] = []
    for column_index in range(len(matched_layout[0])):
        column_signal = np.zeros_like(signal)
        lead_indices = [row[column_index] for row in matched_layout]
        column_signal[lead_indices] = signal[lead_indices]
        column_signals.append(column_signal)
    return column_signals


def _aggregate_probability_vectors(
    probability_vectors: list[np.ndarray],
    method: str,
) -> np.ndarray:
    """Aggregate independent paper-column diagnosis probabilities."""
    if not probability_vectors:
        raise ValueError("At least one probability vector is required")

    stacked = np.stack(probability_vectors)
    if method == "mean":
        return cast(np.ndarray, np.mean(stacked, axis=0))
    if method == "max":
        return cast(np.ndarray, np.max(stacked, axis=0))
    raise ValueError(f"Unsupported probability aggregation: {method}")


def _extract_segment_ensemble_probabilities(
    diagnoser: ECGDiagnoser,
    signal: np.ndarray,
    *,
    layout_name: str,
    aggregation: str,
) -> np.ndarray:
    """Diagnose each printed paper column independently and aggregate outputs."""
    column_probabilities = [
        _extract_probabilities(diagnoser, column_signal)
        for column_signal in _build_layout_column_signals(signal, layout_name)
    ]
    return _aggregate_probability_vectors(column_probabilities, aggregation)


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


def _classification_metric_deltas(
    baseline: dict,
    roundtrip: dict,
) -> dict[str, float]:
    """Return round-trip minus matched-baseline classification metrics."""
    metric_names = (
        "macro_auroc",
        "macro_average_precision",
        "micro_f1",
        "macro_f1",
    )
    return {
        name: round(float(roundtrip[name]) - float(baseline[name]), 6)
        for name in metric_names
    }


def _build_diagnosis_comparison(
    *,
    probabilities: list[np.ndarray],
    successful_ids: list[int],
    baseline_probs: dict[int, np.ndarray],
    scp_codes_by_id: dict[int, dict[str, float]],
    filenames_by_id: dict[int, str],
    threshold: float,
    official_labels_path: Path,
) -> dict:
    """Build baseline-agreement and matched-ground-truth metrics for one method."""
    if not probabilities:
        return {"n_successful": 0}

    probability_matrix = np.stack(probabilities)
    matched_baseline_matrix = np.stack(
        [baseline_probs[ecg_id] for ecg_id in successful_ids]
    )
    cosine_similarities = [
        _cosine_similarity(baseline, candidate)
        for baseline, candidate in zip(
            matched_baseline_matrix, probability_matrix, strict=True
        )
    ]
    abs_prob_diffs = np.mean(
        np.abs(matched_baseline_matrix - probability_matrix),
        axis=1,
    )
    agreement_rates = [
        _agreement_rate(baseline, candidate, threshold)
        for baseline, candidate in zip(
            matched_baseline_matrix, probability_matrix, strict=True
        )
    ]

    result: dict = {
        "n_successful": len(successful_ids),
        "mean_cosine_similarity_vs_baseline": round(
            float(np.mean(cosine_similarities)), 4
        ),
        "mean_abs_prob_diff_vs_baseline": round(float(np.mean(abs_prob_diffs)), 6),
        "agreement_rate_vs_baseline": round(float(np.mean(agreement_rates)), 4),
    }

    matched_scp_codes = [scp_codes_by_id[ecg_id] for ecg_id in successful_ids]
    candidate_semantic = build_ground_truth_evaluation(
        matched_scp_codes,
        probability_matrix,
        threshold=threshold,
    )
    matched_baseline_semantic = build_ground_truth_evaluation(
        matched_scp_codes,
        matched_baseline_matrix,
        threshold=threshold,
    )
    result["semantic_scp_subset_evaluation"] = candidate_semantic
    result["matched_baseline_semantic_scp_subset_evaluation"] = matched_baseline_semantic
    result["semantic_scp_metric_delta_vs_matched_baseline"] = (
        _classification_metric_deltas(
            matched_baseline_semantic["classification"],
            candidate_semantic["classification"],
        )
    )

    if official_labels_path.exists():
        matched_filenames = [filenames_by_id[ecg_id] for ecg_id in successful_ids]
        candidate_official = build_official_ground_truth_evaluation(
            matched_filenames,
            probability_matrix,
            official_labels_path,
            threshold=threshold,
        )
        matched_baseline_official = build_official_ground_truth_evaluation(
            matched_filenames,
            matched_baseline_matrix,
            official_labels_path,
            threshold=threshold,
        )
        result["official_ground_truth_evaluation"] = candidate_official
        result["matched_baseline_official_ground_truth_evaluation"] = matched_baseline_official
        result["official_metric_delta_vs_matched_baseline"] = (
            _classification_metric_deltas(
                matched_baseline_official["classification"],
                candidate_official["classification"],
            )
        )

    return result


def load_test_fold_ids(data_dir: Path, max_samples: int | None) -> list[tuple[int, str]]:
    """Load PTB-XL test fold ecg_ids and their file paths.

    Args:
        data_dir: Path to PTB-XL dataset root.
        max_samples: Optional limit on number of records.

    Returns:
        List of (ecg_id, record_path) tuples for test fold records.
    """
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
    official_labels_path: Path = DEFAULT_OFFICIAL_LABELS_PATH,
    paper_layout: str = DEFAULT_PAPER_LAYOUT,
    segment_aggregation: str = DEFAULT_SEGMENT_AGGREGATION,
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
    if not any(layout_name in paper_layout for layout_name in PAPER_LAYOUTS):
        raise ValueError(f"Unsupported paper layout: {paper_layout}")
    if segment_aggregation not in {"mean", "max"}:
        raise ValueError(
            f"Unsupported probability aggregation: {segment_aggregation}"
        )

    records = load_test_fold_ids(data_dir, max_samples)
    metadata = load_ptbxl_metadata(data_dir)
    test_metadata = metadata[metadata.strat_fold == 10]
    if max_samples is not None:
        test_metadata = test_metadata.head(max_samples)
    scp_codes_by_id = {
        int(ecg_id): row.scp_codes
        for ecg_id, row in test_metadata.iterrows()
    }
    filenames_by_id = {
        int(ecg_id): str(row.filename_hr)
        for ecg_id, row in test_metadata.iterrows()
    }
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
        "rate_adjustments_applied": False,
        "official_labels_available": official_labels_path.exists(),
        "segment_ensemble": {
            "paper_layout": paper_layout,
            "aggregation": segment_aggregation,
        },
        "scenarios": {
            "baseline": {
                "description": "Clean PTB-XL WFDB signals",
                "n_successful": len(baseline_probs),
                "n_failed": baseline_failures,
                "mean_probability": float(np.mean(np.stack(baseline_prob_values)))
                if baseline_prob_values else 0.0,
            },
        },
    }
    if baseline_probs:
        baseline_ids = [ecg_id for ecg_id, _ in records if ecg_id in baseline_probs]
        baseline_matrix = np.stack([baseline_probs[ecg_id] for ecg_id in baseline_ids])
        results["scenarios"]["baseline"]["semantic_scp_subset_evaluation"] = (
            build_ground_truth_evaluation(
                [scp_codes_by_id[ecg_id] for ecg_id in baseline_ids],
                baseline_matrix,
                threshold=threshold,
            )
        )
        if official_labels_path.exists():
            results["scenarios"]["baseline"]["official_ground_truth_evaluation"] = (
                build_official_ground_truth_evaluation(
                    [filenames_by_id[ecg_id] for ecg_id in baseline_ids],
                    baseline_matrix,
                    official_labels_path,
                    threshold=threshold,
                )
            )

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
        roundtrip_probabilities: list[np.ndarray] = []
        successful_ids: list[int] = []
        segment_ensemble_probabilities: list[np.ndarray] = []
        segment_ensemble_successful_ids: list[int] = []
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
                roundtrip_probabilities.append(roundtrip_probs)
                successful_ids.append(ecg_id)

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
                continue

            try:
                segment_ensemble_probs = _extract_segment_ensemble_probabilities(
                    diagnoser,
                    digitized_signal,
                    layout_name=paper_layout,
                    aggregation=segment_aggregation,
                )
                segment_ensemble_probabilities.append(segment_ensemble_probs)
                segment_ensemble_successful_ids.append(ecg_id)
            except Exception as error:
                print(
                    "  Warning: "
                    f"{scenario_name} segment ensemble failed for ecg_id={ecg_id}: {error}"
                )

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
            roundtrip_matrix = np.stack(roundtrip_probabilities)
            matched_baseline_matrix = np.stack(
                [baseline_probs[ecg_id] for ecg_id in successful_ids]
            )
            matched_scp_codes = [scp_codes_by_id[ecg_id] for ecg_id in successful_ids]
            roundtrip_semantic = build_ground_truth_evaluation(
                matched_scp_codes,
                roundtrip_matrix,
                threshold=threshold,
            )
            matched_baseline_semantic = build_ground_truth_evaluation(
                matched_scp_codes,
                matched_baseline_matrix,
                threshold=threshold,
            )
            scenario_result["semantic_scp_subset_evaluation"] = roundtrip_semantic
            scenario_result["matched_baseline_semantic_scp_subset_evaluation"] = (
                matched_baseline_semantic
            )
            scenario_result["semantic_scp_metric_delta_vs_matched_baseline"] = (
                _classification_metric_deltas(
                    matched_baseline_semantic["classification"],
                    roundtrip_semantic["classification"],
                )
            )
            if official_labels_path.exists():
                matched_filenames = [filenames_by_id[ecg_id] for ecg_id in successful_ids]
                roundtrip_official = build_official_ground_truth_evaluation(
                    matched_filenames,
                    roundtrip_matrix,
                    official_labels_path,
                    threshold=threshold,
                )
                matched_baseline_official = build_official_ground_truth_evaluation(
                    matched_filenames,
                    matched_baseline_matrix,
                    official_labels_path,
                    threshold=threshold,
                )
                scenario_result["official_ground_truth_evaluation"] = roundtrip_official
                scenario_result["matched_baseline_official_ground_truth_evaluation"] = (
                    matched_baseline_official
                )
                scenario_result["official_metric_delta_vs_matched_baseline"] = (
                    _classification_metric_deltas(
                        matched_baseline_official["classification"],
                        roundtrip_official["classification"],
                    )
                )
        scenario_result["segment_ensemble"] = {
            "description": (
                f"Independent {paper_layout} paper-column inference with "
                f"{segment_aggregation} probability aggregation"
            ),
            **_build_diagnosis_comparison(
                probabilities=segment_ensemble_probabilities,
                successful_ids=segment_ensemble_successful_ids,
                baseline_probs=baseline_probs,
                scp_codes_by_id=scp_codes_by_id,
                filenames_by_id=filenames_by_id,
                threshold=threshold,
                official_labels_path=official_labels_path,
            ),
        }

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

    segment_config = results.get("segment_ensemble", {})
    print("-" * 75)
    print(
        "SEGMENT ENSEMBLE "
        f"(layout={segment_config.get('paper_layout')}, "
        f"aggregation={segment_config.get('aggregation')})"
    )
    print(f"{'Scenario':<22} {'N':>5} {'CosSim':>8} {'AbsDiff':>9} {'Agree%':>8}")
    print("-" * 75)
    for name, data in results["scenarios"].items():
        if name == "baseline":
            continue
        segment = data.get("segment_ensemble", {})
        n = segment.get("n_successful", 0)
        cos_sim = segment.get("mean_cosine_similarity_vs_baseline", "—")
        abs_diff = segment.get("mean_abs_prob_diff_vs_baseline", "—")
        agree = segment.get("agreement_rate_vs_baseline", "—")
        cos_str = f"{cos_sim:.4f}" if isinstance(cos_sim, float) else cos_sim
        diff_str = f"{abs_diff:.6f}" if isinstance(abs_diff, float) else abs_diff
        agree_str = f"{agree:.4f}" if isinstance(agree, float) else agree
        print(f"{name:<22} {n:>5} {cos_str:>8} {diff_str:>9} {agree_str:>8}")

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
    parser.add_argument(
        "--official-labels", type=Path, default=DEFAULT_OFFICIAL_LABELS_PATH,
        help=f"Official ECGFounder PTB-XL label CSV (default: {DEFAULT_OFFICIAL_LABELS_PATH})",
    )
    parser.add_argument(
        "--paper-layout",
        choices=tuple(PAPER_LAYOUTS),
        default=DEFAULT_PAPER_LAYOUT,
        help=f"Paper layout for segment-aware inference (default: {DEFAULT_PAPER_LAYOUT})",
    )
    parser.add_argument(
        "--segment-aggregation",
        choices=("mean", "max"),
        default=DEFAULT_SEGMENT_AGGREGATION,
        help=(
            "Probability aggregation for independent paper columns "
            f"(default: {DEFAULT_SEGMENT_AGGREGATION})"
        ),
    )
    args = parser.parse_args()

    results = evaluate_roundtrip(
        data_dir=args.data_dir,
        digitized_dir=args.digitized_dir,
        model_path=args.model,
        threshold=args.threshold,
        max_samples=args.max_samples,
        official_labels_path=args.official_labels,
        paper_layout=args.paper_layout,
        segment_aggregation=args.segment_aggregation,
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
