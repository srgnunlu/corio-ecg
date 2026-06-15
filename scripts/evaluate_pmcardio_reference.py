"""Evaluate Corio digitization against matched PMcardio printed segments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_real_photos import (  # noqa: E402
    is_reusable_record,
    run_isolated_attempt,
    sha256_file,
)
from src.training.reference_fidelity import (  # noqa: E402
    evaluate_absolute_printed_segments,
    evaluate_printed_segments,
)

DEFAULT_DATA_DIR = Path("data/reference/pmcardio")
DEFAULT_OUTPUT_DIR = Path("results/pmcardio-reference")
DEFAULT_TIMEOUT_SECONDS = 180
VALID_MODES = ("default", "retry", "perspective", "shadow", "layout_segments")
MAX_SHIFT_SAMPLES = 50  # 100 ms at 500 Hz
BOOTSTRAP_SEED = 20260613
BOOTSTRAP_RESAMPLES = 2000


def _experiment_signal_paths(
    output_dir: Path,
    *,
    category: str,
    image_stem: str,
    mode: str,
) -> tuple[Path, Path]:
    """Return isolated signal paths for a non-default experiment."""
    if mode == "default":
        raise ValueError("default mode uses the frozen baseline signal paths")
    return (
        output_dir / "signals" / category / f"{image_stem}.npy",
        output_dir / "signals-calibrated" / category / f"{image_stem}.npy",
    )


def _bootstrap_ci95(
    values: list[float],
    *,
    statistic: str,
) -> dict[str, float | int] | None:
    """Return a deterministic percentile bootstrap confidence interval."""
    if not values:
        return None
    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    if len(values) == 1:
        value = values[0]
        return {"low": value, "high": value, "resamples": BOOTSTRAP_RESAMPLES}

    samples = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(samples), size=(BOOTSTRAP_RESAMPLES, len(samples)))
    resampled = samples[indices]
    estimates = (
        np.mean(resampled, axis=1)
        if statistic == "mean"
        else np.median(resampled, axis=1)
    )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "low": float(low),
        "high": float(high),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _aggregate_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        record
        for record in records
        if record.get("status") == "success" and record.get("fidelity") is not None
    ]
    correlations = [
        float(record["fidelity"]["median_correlation"])
        for record in successful
    ]
    absolute = [
        record["fidelity"]["absolute"]
        for record in successful
        if record["fidelity"].get("absolute") is not None
    ]
    rmse_values = [float(metrics["median_rmse_mv"]) for metrics in absolute]
    snr_values = [float(metrics["median_snr_db"]) for metrics in absolute]
    gain_values = [float(metrics["median_gain_ratio"]) for metrics in absolute]
    return {
        "total": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "median_correlation": statistics.median(correlations) if correlations else None,
        "mean_correlation": statistics.mean(correlations) if correlations else None,
        "median_correlation_ci95": _bootstrap_ci95(correlations, statistic="median"),
        "mean_correlation_ci95": _bootstrap_ci95(correlations, statistic="mean"),
        "absolute_amplitude": {
            "available": len(absolute),
            "median_rmse_mv": statistics.median(rmse_values) if rmse_values else None,
            "median_rmse_mv_ci95": _bootstrap_ci95(rmse_values, statistic="median"),
            "median_snr_db": statistics.median(snr_values) if snr_values else None,
            "median_snr_db_ci95": _bootstrap_ci95(snr_values, statistic="median"),
            "median_gain_ratio": statistics.median(gain_values) if gain_values else None,
            "median_gain_ratio_ci95": _bootstrap_ci95(gain_values, statistic="median"),
        },
    }


def aggregate_fidelity_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate matched fidelity results overall and by capture category."""
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


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    fidelity = record.get("fidelity") or {}
    absolute = fidelity.get("absolute") or {}
    diagnostics = record.get("diagnostics") or {}
    return {
        "category": record.get("category"),
        "image_id": record.get("image_id"),
        "image_path": record.get("image_path"),
        "ecg_id": record.get("ecg_id"),
        "layout": record.get("layout"),
        "split": record.get("split"),
        "status": record.get("status"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "error": record.get("error"),
        "layout_name": diagnostics.get("layout_name"),
        "nonzero_leads_count": diagnostics.get("nonzero_leads_count"),
        "einthoven_score": diagnostics.get("einthoven_score"),
        "median_correlation": fidelity.get("median_correlation"),
        "mean_correlation": fidelity.get("mean_correlation"),
        "minimum_correlation": fidelity.get("minimum_correlation"),
        "median_normalized_rmse": fidelity.get("median_normalized_rmse"),
        "median_rmse_mv": absolute.get("median_rmse_mv"),
        "median_snr_db": absolute.get("median_snr_db"),
        "median_gain_ratio": absolute.get("median_gain_ratio"),
    }


def write_report(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    selection_manifest: dict[str, Any] | None = None,
    selected_split: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pmcardio_reference_fidelity.json"
    csv_path = output_dir / "pmcardio_reference_fidelity.csv"
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "name": "PMcardio ECG Image Database (PM-ECG-ID)",
            "doi": "10.5281/zenodo.13617673",
            "url": "https://zenodo.org/records/13617673",
            "license": "GPL-3.0-or-later",
        },
        "metric": {
            "primary": "median per-lead shifted correlation",
            "alignment": "up to +/-100 ms per lead",
            "limitations": (
                "Absolute metrics use the digitizer's pixel-derived calibration before "
                "global z-score normalization. They measure reconstruction fidelity, "
                "not clinical diagnostic accuracy."
            ),
        },
        "selection_manifest": selection_manifest,
        "selected_split": selected_split,
        "aggregate": aggregate_fidelity_records(records),
        "records": records,
    }
    json_path.write_text(json.dumps(report, indent=2))

    rows = [_flatten_record(record) for record in records]
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def select_evaluation_metadata(
    metadata: pd.DataFrame,
    selected_split: str | None,
) -> pd.DataFrame:
    """Require explicit split selection for pre-registered holdout metadata."""
    if "split" not in metadata.columns:
        if selected_split is not None:
            raise ValueError("selected split requested but metadata has no split column")
        return metadata
    if selected_split not in {"tune", "test"}:
        raise ValueError("pre-registered holdout evaluation requires --split tune or test")
    selected = metadata[metadata["split"] == selected_split].copy()
    if selected.empty:
        raise ValueError(f"selected split contains no records: {selected_split}")
    return selected.reset_index(drop=True)


def evaluate_subset(
    data_dir: Path,
    output_dir: Path,
    *,
    timeout_seconds: int,
    overwrite: bool,
    max_images: int | None,
    mode: str = "default",
    selected_split: str | None = None,
) -> list[dict[str, Any]]:
    metadata = pd.read_csv(data_dir / "subset_metadata.csv")
    metadata = select_evaluation_metadata(metadata, selected_split)
    if max_images is not None:
        metadata = metadata.head(max_images)
    references = np.load(data_dir / "leads.npz")
    records: list[dict[str, Any]] = []

    for index, row in metadata.iterrows():
        relative_path = str(row["Image relative path"])
        category = str(row["category"])
        layout = str(row["ECG format"])
        image_path = data_dir / "images" / relative_path
        record_id = f"{category}__{image_path.stem}"
        if mode == "default":
            signal_path = data_dir / "digitized" / category / f"{image_path.stem}.npy"
            calibrated_path = (
                data_dir / "digitized-calibrated" / category / f"{image_path.stem}.npy"
            )
            operational_path = output_dir / "operational" / f"{record_id}.json"
        else:
            signal_path, calibrated_path = _experiment_signal_paths(
                output_dir,
                category=category,
                image_stem=image_path.stem,
                mode=mode,
            )
            operational_path = output_dir / "operational" / mode / f"{record_id}.json"
        source_sha256 = sha256_file(image_path)

        print(f"{index + 1}/{len(metadata)} {relative_path}", flush=True)
        reusable = (
            not overwrite
            and signal_path.exists()
            and calibrated_path.exists()
            and is_reusable_record(
                operational_path,
                mode=mode,
                source_sha256=source_sha256,
                layout_hint=layout,
            )
        )
        if reusable:
            operational = json.loads(operational_path.read_text())
        else:
            operational = run_isolated_attempt(
                image_path,
                mode=mode,
                layout_hint=layout,
                signal_path=signal_path,
                record_path=operational_path,
                timeout_seconds=timeout_seconds,
                calibrated_signal_path=calibrated_path,
            )

        record: dict[str, Any] = {
            **operational,
            "category": category,
            "image_id": int(row["Image ID"]),
            "image_path": relative_path,
            "ecg_id": str(row["ECG ID"]),
            "layout": layout,
            "split": row.get("split"),
            "reference_key": str(row["reference_key"]),
            "fidelity": None,
        }
        if operational.get("status") == "success":
            reference = references[str(row["reference_key"])]
            digitized = np.load(signal_path)
            record["fidelity"] = evaluate_printed_segments(
                reference,
                digitized,
                max_shift_samples=MAX_SHIFT_SAMPLES,
            )
            if calibrated_path.exists():
                calibrated = np.load(calibrated_path)
                record["fidelity"]["absolute"] = evaluate_absolute_printed_segments(
                    reference,
                    calibrated,
                    max_shift_samples=MAX_SHIFT_SAMPLES,
                )
        records.append(record)

    references.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode", choices=VALID_MODES, default="default")
    parser.add_argument("--split", choices=("tune", "test"))
    args = parser.parse_args()

    records = evaluate_subset(
        args.data_dir,
        args.output_dir,
        timeout_seconds=args.timeout,
        overwrite=args.overwrite,
        max_images=args.max_images,
        mode=args.mode,
        selected_split=args.split,
    )
    manifest_path = args.data_dir / "selection_manifest.json"
    selection_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    json_path, _ = write_report(
        records,
        args.output_dir,
        selection_manifest=selection_manifest,
        selected_split=args.split,
    )
    aggregate = aggregate_fidelity_records(records)
    print(f"\nReport: {json_path}")
    print(
        f"Success: {aggregate['successful']}/{aggregate['total']} "
        f"({aggregate['success_rate']:.1%})"
    )
    print(f"Median correlation: {aggregate['median_correlation']}")


if __name__ == "__main__":
    main()
