"""Compare ECGFounder outputs on matched PMcardio reference and digitized signals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_pmcardio_reference import _bootstrap_ci95  # noqa: E402
from src.pipeline.diagnose import ECGDiagnoser  # noqa: E402
from src.training.evaluate_roundtrip import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    _agreement_rate,
    _cosine_similarity,
    _extract_probabilities,
    _extract_segment_ensemble_probabilities,
)

DEFAULT_DATA_DIR = Path("data/reference/pmcardio")
DEFAULT_OUTPUT_DIR = Path("results/pmcardio-reference")
DEFAULT_THRESHOLD = 0.5
TARGET_LENGTH = 5000
METHODS = ("tiled", "segment_ensemble")


def prepare_reference_model_input(reference: np.ndarray) -> np.ndarray:
    """Tile matched printed segments into a normalized ECGFounder input."""
    reference = np.asarray(reference)
    if reference.ndim != 2 or reference.shape[1] != 12:
        raise ValueError("reference must have shape (printed_samples, 12)")
    signal = reference.T.astype(np.float64)
    repeats = (TARGET_LENGTH + signal.shape[1] - 1) // signal.shape[1]
    tiled = np.tile(signal, (1, repeats))[:, :TARGET_LENGTH]
    normalized = (tiled - float(np.mean(tiled))) / (float(np.std(tiled)) + 1e-8)
    return normalized.astype(np.float32)


def compare_probability_vectors(
    reference: np.ndarray,
    digitized: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Measure diagnosis-output consistency for one matched signal pair."""
    return {
        "cosine_similarity": _cosine_similarity(reference, digitized),
        "mean_abs_probability_difference": float(np.mean(np.abs(reference - digitized))),
        "agreement_rate": _agreement_rate(reference, digitized, threshold),
    }


def _aggregate_method(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    metrics = [
        record["methods"][method]
        for record in records
        if record.get("status") == "success" and method in record.get("methods", {})
    ]
    cosine = [float(item["cosine_similarity"]) for item in metrics]
    abs_diff = [float(item["mean_abs_probability_difference"]) for item in metrics]
    agreement = [float(item["agreement_rate"]) for item in metrics]
    return {
        "n": len(metrics),
        "mean_cosine_similarity": float(np.mean(cosine)) if cosine else None,
        "median_cosine_similarity": float(np.median(cosine)) if cosine else None,
        "mean_cosine_similarity_ci95": _bootstrap_ci95(cosine, statistic="mean"),
        "mean_abs_probability_difference": float(np.mean(abs_diff)) if abs_diff else None,
        "mean_agreement_rate": float(np.mean(agreement)) if agreement else None,
        "mean_agreement_rate_ci95": _bootstrap_ci95(agreement, statistic="mean"),
    }


def _aggregate_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    return {
        "total": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "methods": {method: _aggregate_method(records, method) for method in METHODS},
    }


def aggregate_drift_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate diagnosis consistency overall and by physical category."""
    categories = sorted({str(record["category"]) for record in records})
    return {
        **_aggregate_subset(records),
        "categories": {
            category: _aggregate_subset(
                [record for record in records if record["category"] == category]
            )
            for category in categories
        },
    }


def _probabilities_for_method(
    diagnoser: ECGDiagnoser,
    signal: np.ndarray,
    *,
    method: str,
    layout: str,
) -> np.ndarray:
    if method == "tiled":
        return _extract_probabilities(diagnoser, signal)
    if method == "segment_ensemble":
        return _extract_segment_ensemble_probabilities(
            diagnoser,
            signal,
            layout_name=layout,
            aggregation="mean",
        )
    raise ValueError(f"Unsupported diagnosis drift method: {method}")


def evaluate_diagnosis_drift(
    data_dir: Path,
    model_path: Path,
    *,
    threshold: float,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate matched-reference diagnosis consistency for the selected subset."""
    metadata = pd.read_csv(data_dir / "subset_metadata.csv")
    if max_images is not None:
        metadata = metadata.head(max_images)
    references = np.load(data_dir / "leads.npz")
    diagnoser = ECGDiagnoser(checkpoint_path=model_path)
    reference_cache: dict[tuple[str, str, str], np.ndarray] = {}
    records: list[dict[str, Any]] = []

    for index, row in metadata.iterrows():
        category = str(row["category"])
        image_path = str(row["Image relative path"])
        image_stem = Path(image_path).stem
        layout = str(row["ECG format"])
        reference_key = str(row["reference_key"])
        signal_path = data_dir / "digitized" / category / f"{image_stem}.npy"
        print(f"{index + 1}/{len(metadata)} {image_path}", flush=True)
        record: dict[str, Any] = {
            "category": category,
            "image_id": int(row["Image ID"]),
            "image_path": image_path,
            "ecg_id": str(row["ECG ID"]),
            "reference_key": reference_key,
            "layout": layout,
            "status": "failed",
            "error": None,
            "methods": {},
        }
        try:
            reference_input = prepare_reference_model_input(references[reference_key])
            digitized_input = np.load(signal_path)
            for method in METHODS:
                cache_key = (reference_key, layout, method)
                if cache_key not in reference_cache:
                    reference_cache[cache_key] = _probabilities_for_method(
                        diagnoser,
                        reference_input,
                        method=method,
                        layout=layout,
                    )
                digitized_probs = _probabilities_for_method(
                    diagnoser,
                    digitized_input,
                    method=method,
                    layout=layout,
                )
                record["methods"][method] = compare_probability_vectors(
                    reference_cache[cache_key],
                    digitized_probs,
                    threshold=threshold,
                )
            record["status"] = "success"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)

    references.close()
    return records


def write_report(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    threshold: float,
    selection_manifest: dict[str, Any] | None,
) -> tuple[Path, Path]:
    """Write JSON and flat CSV diagnosis-drift reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pmcardio_diagnosis_drift.json"
    csv_path = output_dir / "pmcardio_diagnosis_drift.csv"
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "threshold": threshold,
        "selection_manifest": selection_manifest,
        "limitations": (
            "This compares ECGFounder output consistency between matched reference "
            "and digitized signals. It is not a clinical-accuracy evaluation."
        ),
        "aggregate": aggregate_drift_records(records),
        "records": records,
    }
    json_path.write_text(json.dumps(report, indent=2))

    rows: list[dict[str, Any]] = []
    for record in records:
        for method in METHODS:
            metrics = record.get("methods", {}).get(method, {})
            rows.append(
                {
                    "category": record.get("category"),
                    "image_id": record.get("image_id"),
                    "image_path": record.get("image_path"),
                    "ecg_id": record.get("ecg_id"),
                    "layout": record.get("layout"),
                    "status": record.get("status"),
                    "error": record.get("error"),
                    "method": method,
                    **metrics,
                }
            )
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    records = evaluate_diagnosis_drift(
        args.data_dir,
        args.model,
        threshold=args.threshold,
        max_images=args.max_images,
    )
    manifest_path = args.data_dir / "selection_manifest.json"
    selection_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    json_path, _ = write_report(
        records,
        args.output_dir,
        threshold=args.threshold,
        selection_manifest=selection_manifest,
    )
    aggregate = aggregate_drift_records(records)
    print(f"\nReport: {json_path}")
    print(f"Success: {aggregate['successful']}/{aggregate['total']}")
    for method, metrics in aggregate["methods"].items():
        print(
            f"{method}: cosine={metrics['mean_cosine_similarity']:.4f}, "
            f"agreement={metrics['mean_agreement_rate']:.2%}, "
            f"abs_diff={metrics['mean_abs_probability_difference']:.4f}"
        )


if __name__ == "__main__":
    main()
