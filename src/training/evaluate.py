# Baseline evaluation script for PTB-XL — runs ECGFounder on test fold and saves metrics

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pipeline.diagnose import ECGDiagnoser
from src.training.ptbxl_labels import (
    MAPPED_ECGFOUNDER_INDICES,
    MAPPED_ECGFOUNDER_LABELS,
    SCP_TO_ECGFOUNDER_LABEL,
    build_ground_truth_matrix,
)
from src.utils.ecg_labels import ECG_FOUNDER_LABELS, NUM_CLASSES
from src.utils.metrics import compute_multilabel_classification_metrics
from src.utils.wfdb_helpers import read_ecg_signal

DEFAULT_MODEL_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")
DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_OUTPUT_DIR = Path("results/metrics")
DEFAULT_OFFICIAL_LABELS_PATH = DEFAULT_DATA_DIR / "ecgfounder_ptbxl_label.csv"
ECGFOUNDER_OFFICIAL_EVAL_COMMIT = "04edac702b61c91face519774ddcc0cd712fef23"

# PTB-XL convention: fold 10 is the held-out test fold, fold 9 the validation
# fold. Calibration must be fitted on 9 so that 10 stays a clean final audit.
DEFAULT_TEST_FOLD = 10
DEFAULT_CALIBRATION_FOLD = 9


def build_ground_truth_evaluation(
    records_scp_codes: list[dict[str, float]],
    probability_matrix: np.ndarray,
    threshold: float,
    minimum_positive_examples: int = 5,
) -> dict:
    """Compare probabilities with the explicit semantic SCP-code subset."""
    ground_truth = build_ground_truth_matrix(records_scp_codes)
    mapped_ground_truth = ground_truth[:, MAPPED_ECGFOUNDER_INDICES]
    mapped_probabilities = probability_matrix[:, MAPPED_ECGFOUNDER_INDICES]
    classification = compute_multilabel_classification_metrics(
        mapped_ground_truth,
        mapped_probabilities,
        class_names=MAPPED_ECGFOUNDER_LABELS,
        threshold=threshold,
        minimum_positive_examples=minimum_positive_examples,
    )
    return {
        "mapping_version": 1,
        "mapped_scp_codes": len(SCP_TO_ECGFOUNDER_LABEL),
        "mapped_model_outputs": len(MAPPED_ECGFOUNDER_INDICES),
        "records_with_mapped_labels": int(np.any(mapped_ground_truth, axis=1).sum()),
        "mapped_label_assignments": int(mapped_ground_truth.sum()),
        "classification": classification,
    }


def build_official_ground_truth_evaluation(
    record_filenames: list[str],
    probability_matrix: np.ndarray,
    labels_path: Path,
    threshold: float,
    minimum_positive_examples: int = 5,
) -> dict:
    """Compare probabilities with ECGFounder's official PTB-XL target vectors."""
    labels = pd.read_csv(labels_path, usecols=["filename_hr", "label"])
    if labels.filename_hr.duplicated().any():
        raise ValueError("Official ECGFounder label CSV contains duplicate filenames")

    parsed_labels = {
        str(row.filename_hr): np.asarray(json.loads(row.label), dtype=np.int8)
        for row in labels.itertuples(index=False)
    }
    invalid_lengths = {
        filename: len(label)
        for filename, label in parsed_labels.items()
        if len(label) != NUM_CLASSES
    }
    if invalid_lengths:
        raise ValueError("Official ECGFounder label vectors must contain 150 values")

    matched_indices = [
        index for index, filename in enumerate(record_filenames)
        if filename in parsed_labels
    ]
    matched_ground_truth = np.stack(
        [parsed_labels[record_filenames[index]] for index in matched_indices],
        axis=0,
    ) if matched_indices else np.empty((0, NUM_CLASSES), dtype=np.int8)
    matched_probabilities = probability_matrix[matched_indices]

    classification = compute_multilabel_classification_metrics(
        matched_ground_truth,
        matched_probabilities,
        class_names=ECG_FOUNDER_LABELS,
        threshold=threshold,
        minimum_positive_examples=minimum_positive_examples,
    )
    return {
        "source": "PKUDigitalHealth/ECGFounder csv/ptbxl_label.csv",
        "source_commit": ECGFOUNDER_OFFICIAL_EVAL_COMMIT,
        "matched_records": len(matched_indices),
        "records_missing_labels": len(record_filenames) - len(matched_indices),
        "classification": classification,
    }


def load_ptbxl_metadata(data_dir: Path) -> pd.DataFrame:
    """Load PTB-XL metadata CSV and parse scp_codes column.

    Args:
        data_dir: Path to PTB-XL dataset root directory.

    Returns:
        DataFrame with parsed scp_codes column indexed by ecg_id.
    """
    csv_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(csv_path, index_col="ecg_id")

    # ast.literal_eval is safe — only parses Python literals
    # PTB-XL stores scp_codes as Python dict strings like {'NORM': 100.0}
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    return df


def evaluate_ptbxl(
    data_dir: Path,
    model_path: Path,
    max_samples: int | None = None,
    threshold: float = 0.5,
    minimum_positive_examples: int = 5,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    official_labels_path: Path = DEFAULT_OFFICIAL_LABELS_PATH,
    fold: int = DEFAULT_TEST_FOLD,
    output_prefix: str = "ptbxl_baseline",
) -> dict:
    """Run ECGFounder evaluation on one PTB-XL stratified fold.

    Args:
        data_dir: Path to PTB-XL dataset root directory.
        model_path: Path to ECGFounder checkpoint file.
        max_samples: Optional limit on number of records to evaluate.
        fold: PTB-XL strat_fold to evaluate. Fold 10 is the held-out test
            fold; fold 9 is the calibration/validation fold.
        output_prefix: Basename prefix for saved artefacts, so calibration
            runs never overwrite the locked test-fold baseline.

    Returns:
        Summary dictionary with evaluation statistics.
    """
    metadata = load_ptbxl_metadata(data_dir)
    test_records = metadata[metadata.strat_fold == fold]

    if max_samples is not None:
        test_records = test_records.head(max_samples)

    print(f"Evaluating {len(test_records)} records from PTB-XL fold {fold}...")

    # Load the diagnosis model
    diagnoser = ECGDiagnoser(checkpoint_path=model_path)

    # Collect all 150-class probabilities for each record
    all_probabilities: list[np.ndarray] = []
    successful_scp_codes: list[dict[str, float]] = []
    successful_record_ids: list[int] = []
    successful_filenames: list[str] = []

    for ecg_id, row in tqdm(test_records.iterrows(), total=len(test_records)):
        try:
            # Build record path — remove .hea extension if present
            record_path = str(data_dir / row.filename_hr)
            if record_path.endswith(".hea"):
                record_path = record_path[:-4]

            signal, _ = read_ecg_signal(record_path)
            results = diagnoser.diagnose_all(signal, apply_rate_adjustments=False)

            # Convert sorted results back to ordered probability array
            probabilities = np.zeros(NUM_CLASSES, dtype=np.float32)
            for result in results:
                probabilities[result.index] = result.probability

            all_probabilities.append(probabilities)
            successful_scp_codes.append(row.scp_codes)
            successful_record_ids.append(int(ecg_id))
            successful_filenames.append(str(row.filename_hr))

        except Exception as error:
            print(f"Warning: Failed to process ecg_id={ecg_id}: {error}")

    # Stack into matrix (n_records, 150)
    probability_matrix = (
        np.stack(all_probabilities, axis=0)
        if all_probabilities
        else np.empty((0, NUM_CLASSES), dtype=np.float32)
    )

    # Save probability matrix and row IDs so outputs remain traceable.
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"{output_prefix}_probabilities.npy"
    np.save(npy_path, probability_matrix)
    print(f"Saved probability matrix to {npy_path}")
    np.save(output_dir / f"{output_prefix}_record_ids.npy", np.array(successful_record_ids))

    # Build summary statistics
    probability_stats = {
        "mean": float(np.mean(probability_matrix)) if probability_matrix.size else 0.0,
        "std": float(np.std(probability_matrix)) if probability_matrix.size else 0.0,
        "max": float(np.max(probability_matrix)) if probability_matrix.size else 0.0,
        "min": float(np.min(probability_matrix)) if probability_matrix.size else 0.0,
    }
    summary = {
        "fold": fold,
        "total_records": len(test_records),
        "successful_records": len(successful_record_ids),
        "failed_records": len(test_records) - len(successful_record_ids),
        "rate_adjustments_applied": False,
        "probability_stats": probability_stats,
        "semantic_scp_subset_evaluation": build_ground_truth_evaluation(
            successful_scp_codes,
            probability_matrix,
            threshold=threshold,
            minimum_positive_examples=minimum_positive_examples,
        ),
    }
    if official_labels_path.exists():
        summary["official_ground_truth_evaluation"] = build_official_ground_truth_evaluation(
            successful_filenames,
            probability_matrix,
            official_labels_path,
            threshold=threshold,
            minimum_positive_examples=minimum_positive_examples,
        )
    else:
        summary["official_ground_truth_evaluation"] = None
        print(
            "Official ECGFounder labels not found; run "
            "python scripts/download_ecgfounder_eval_labels.py"
        )

    json_path = output_dir / f"{output_prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

    return summary


def main() -> None:
    """CLI entry point for PTB-XL baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ECGFounder baseline on PTB-XL test fold"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to PTB-XL dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to ECGFounder checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test records to evaluate (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for F1 metrics (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Metric output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--minimum-positive-examples",
        type=int,
        default=5,
        help="Minimum class support required for summary metrics (default: 5)",
    )
    parser.add_argument(
        "--official-labels",
        type=Path,
        default=DEFAULT_OFFICIAL_LABELS_PATH,
        help=f"Official ECGFounder PTB-XL label CSV (default: {DEFAULT_OFFICIAL_LABELS_PATH})",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=DEFAULT_TEST_FOLD,
        help=(
            f"PTB-XL strat_fold to evaluate (default: {DEFAULT_TEST_FOLD} = test fold; "
            f"use {DEFAULT_CALIBRATION_FOLD} for calibration)"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ptbxl_baseline",
        help="Basename prefix for saved artefacts (default: ptbxl_baseline)",
    )

    args = parser.parse_args()
    summary = evaluate_ptbxl(
        data_dir=args.data_dir,
        model_path=args.model,
        max_samples=args.max_samples,
        threshold=args.threshold,
        minimum_positive_examples=args.minimum_positive_examples,
        output_dir=args.output_dir,
        official_labels_path=args.official_labels,
        fold=args.fold,
        output_prefix=args.output_prefix,
    )

    print("\n=== Evaluation Summary ===")
    print(f"Fold: {summary['fold']}")
    print(f"Total records: {summary['total_records']}")
    print(f"Successful: {summary['successful_records']}")
    stats = summary["probability_stats"]
    print(f"Probability stats — mean: {stats['mean']:.4f}, "
          f"std: {stats['std']:.4f}, "
          f"max: {stats['max']:.4f}, "
          f"min: {stats['min']:.4f}")
    official = summary["official_ground_truth_evaluation"]
    if official is not None:
        classification = official["classification"]
        print(
            "Official 150-class metrics — "
            f"matched records: {official['matched_records']}, "
            f"classes: {classification['evaluated_classes']}, "
            f"macro AUROC: {classification['macro_auroc']:.4f}, "
            f"macro AP: {classification['macro_average_precision']:.4f}, "
            f"micro F1: {classification['micro_f1']:.4f}"
        )
    semantic = summary["semantic_scp_subset_evaluation"]
    classification = semantic["classification"]
    print(
        "Semantic SCP subset metrics — "
        f"mapped records: {semantic['records_with_mapped_labels']}, "
        f"classes: {classification['evaluated_classes']}, "
        f"macro AUROC: {classification['macro_auroc']:.4f}, "
        f"macro AP: {classification['macro_average_precision']:.4f}, "
        f"micro F1: {classification['micro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
