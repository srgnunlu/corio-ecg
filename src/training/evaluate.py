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
from src.utils.ecg_labels import ECG_FOUNDER_LABELS, NUM_CLASSES
from src.utils.metrics import compute_auroc_per_class, compute_macro_auroc
from src.utils.wfdb_helpers import read_ecg_signal

DEFAULT_MODEL_PATH = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")
DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")


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
) -> dict:
    """Run ECGFounder evaluation on PTB-XL test fold.

    Args:
        data_dir: Path to PTB-XL dataset root directory.
        model_path: Path to ECGFounder checkpoint file.
        max_samples: Optional limit on number of records to evaluate.

    Returns:
        Summary dictionary with evaluation statistics.
    """
    # Load metadata and filter to test fold (strat_fold == 10)
    metadata = load_ptbxl_metadata(data_dir)
    test_records = metadata[metadata.strat_fold == 10]

    if max_samples is not None:
        test_records = test_records.head(max_samples)

    print(f"Evaluating {len(test_records)} test records from PTB-XL...")

    # Load the diagnosis model
    diagnoser = ECGDiagnoser(checkpoint_path=model_path)

    # Collect all 150-class probabilities for each record
    all_probabilities: list[np.ndarray] = []
    successful_count = 0

    for ecg_id, row in tqdm(test_records.iterrows(), total=len(test_records)):
        try:
            # Build record path — remove .hea extension if present
            record_path = str(data_dir / row.filename_hr)
            if record_path.endswith(".hea"):
                record_path = record_path[:-4]

            signal, _ = read_ecg_signal(record_path)
            results = diagnoser.diagnose_all(signal)

            # Convert sorted results back to ordered probability array
            probabilities = np.zeros(NUM_CLASSES)
            for result in results:
                probabilities[result.index] = result.probability

            all_probabilities.append(probabilities)
            successful_count += 1

        except Exception as error:
            print(f"Warning: Failed to process ecg_id={ecg_id}: {error}")
            all_probabilities.append(np.zeros(NUM_CLASSES))

    # Stack into matrix (n_records, 150)
    probability_matrix = np.stack(all_probabilities, axis=0)

    # Save probability matrix
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / "ptbxl_baseline_probabilities.npy"
    np.save(npy_path, probability_matrix)
    print(f"Saved probability matrix to {npy_path}")

    # Build summary statistics
    summary = {
        "total_records": len(test_records),
        "successful_records": successful_count,
        "probability_stats": {
            "mean": float(np.mean(probability_matrix)),
            "std": float(np.std(probability_matrix)),
            "max": float(np.max(probability_matrix)),
            "min": float(np.min(probability_matrix)),
        },
    }

    json_path = output_dir / "ptbxl_baseline_summary.json"
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

    args = parser.parse_args()
    summary = evaluate_ptbxl(
        data_dir=args.data_dir,
        model_path=args.model,
        max_samples=args.max_samples,
    )

    print("\n=== Evaluation Summary ===")
    print(f"Total records: {summary['total_records']}")
    print(f"Successful: {summary['successful_records']}")
    stats = summary["probability_stats"]
    print(f"Probability stats — mean: {stats['mean']:.4f}, "
          f"std: {stats['std']:.4f}, "
          f"max: {stats['max']:.4f}, "
          f"min: {stats['min']:.4f}")


if __name__ == "__main__":
    main()
