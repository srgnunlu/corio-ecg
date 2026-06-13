"""Operational-quality evaluation for real phone photographs of paper ECGs.

Photos alone cannot establish signal fidelity or diagnostic accuracy. This
script measures digitization success, lead activity, Einthoven consistency,
layout detection, warnings, processing time, and dewarping-retry behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.pipeline.digitize import ECGDigitiser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = Path("data/real-phone/photos")
DEFAULT_METADATA_PATH = Path("data/real-phone/metadata.csv")
DEFAULT_SIGNAL_DIR = Path("data/real-phone/signals")
DEFAULT_OUTPUT_DIR = Path("results/real-phone")
DEFAULT_TIMEOUT_SECONDS = 180
VALID_MODES = ("default", "retry")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REAL_PHOTO_EVALUATION_VERSION = 2


def _json_default(value: object) -> object:
    """Convert common numeric objects into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON document, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a source photo."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_photo_paths(image_dir: Path, max_samples: int | None = None) -> list[Path]:
    """Collect supported photos in stable filename order."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Real-photo directory not found: {image_dir}")
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No JPG or PNG photos found in {image_dir}")
    return paths if max_samples is None else paths[:max_samples]


def load_layout_hints(metadata_path: Path) -> dict[str, str]:
    """Load known per-case layouts from the optional capture metadata CSV."""
    if not metadata_path.exists():
        return {}
    with metadata_path.open(newline="") as metadata_file:
        rows = csv.DictReader(metadata_file)
        return {
            str(row["case_id"]): str(row["layout"])
            for row in rows
            if row.get("case_id")
            and row.get("layout")
            and str(row["layout"]).lower() not in {"unknown", "auto"}
        }


def is_reusable_record(
    record_path: Path,
    *,
    mode: str,
    source_sha256: str,
    layout_hint: str | None,
) -> bool:
    """Return whether a successful audit record matches the current input."""
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("pipeline_version") == REAL_PHOTO_EVALUATION_VERSION
        and record.get("mode") == mode
        and record.get("source_sha256") == source_sha256
        and record.get("layout_hint") == layout_hint
        and record.get("status") == "success"
    )


def _capture_variant(image_path: Path) -> str:
    """Return the anonymous capture variant encoded after ``__`` in a filename."""
    parts = image_path.stem.split("__", maxsplit=1)
    return parts[1] if len(parts) == 2 else "unknown"


def _case_id(image_path: Path) -> str:
    return image_path.stem.split("__", maxsplit=1)[0]


def _record_variant(record: dict[str, Any]) -> str:
    variant = record.get("capture_variant")
    if isinstance(variant, str):
        return variant
    return _capture_variant(Path(str(record.get("source_image", ""))))


def _base_record(
    image_path: Path,
    mode: str,
    source_sha256: str,
    layout_hint: str | None,
) -> dict[str, Any]:
    """Create common audit fields for a photo attempt."""
    try:
        with Image.open(image_path) as image:
            source_width, source_height = image.size
    except OSError:
        source_width, source_height = 0, 0
    return {
        "pipeline_version": REAL_PHOTO_EVALUATION_VERSION,
        "source_image": str(image_path),
        "source_sha256": source_sha256,
        "source_width": source_width,
        "source_height": source_height,
        "mode": mode,
        "capture_variant": _capture_variant(image_path),
        "layout_hint": layout_hint,
        "status": "failed",
        "elapsed_seconds": 0.0,
        "has_warnings": True,
        "diagnostics": {},
        "error": None,
    }


def run_worker(
    image_path: Path,
    *,
    mode: str,
    layout_hint: str | None,
    signal_path: Path,
    calibrated_signal_path: Path | None = None,
) -> dict[str, Any]:
    """Digitize one photo inside an isolated worker process."""
    source_sha256 = sha256_file(image_path)
    record = _base_record(image_path, mode, source_sha256, layout_hint)
    start = time.monotonic()
    digitiser: ECGDigitiser | None = None
    try:
        digitiser = ECGDigitiser(enable_dewarping_retry=mode == "retry")
        if calibrated_signal_path is None:
            signal = digitiser.digitize(image_path, layout_hint=layout_hint)
        else:
            outputs = digitiser.digitize_with_calibrated(
                image_path,
                layout_hint=layout_hint,
            )
            signal = outputs.model_input
            calibrated_signal_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(calibrated_signal_path, outputs.calibrated_millivolts)
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(signal_path, signal)
        record["status"] = "success"
        record["has_warnings"] = digitiser.last_info.has_warnings
        record["diagnostics"] = asdict(digitiser.last_info)
    except Exception as error:
        if digitiser is not None and hasattr(digitiser, "last_info"):
            record["diagnostics"] = asdict(digitiser.last_info)
        record["error"] = f"{type(error).__name__}: {error}"
        signal_path.unlink(missing_ok=True)
        if calibrated_signal_path is not None:
            calibrated_signal_path.unlink(missing_ok=True)
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
    return record


def _failure_record(
    image_path: Path,
    *,
    mode: str,
    source_sha256: str,
    layout_hint: str | None,
    status: str,
    elapsed_seconds: float,
    error: str,
) -> dict[str, Any]:
    record = _base_record(image_path, mode, source_sha256, layout_hint)
    record["status"] = status
    record["elapsed_seconds"] = elapsed_seconds
    record["error"] = error
    return record


def run_isolated_attempt(
    image_path: Path,
    *,
    mode: str,
    layout_hint: str | None,
    signal_path: Path,
    record_path: Path,
    timeout_seconds: int,
    calibrated_signal_path: Path | None = None,
) -> dict[str, Any]:
    """Run one photo attempt in a subprocess and enforce a hard timeout."""
    source_sha256 = sha256_file(image_path)
    record_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-image",
        str(image_path),
        "--worker-mode",
        mode,
        "--worker-output",
        str(record_path),
        "--worker-signal",
        str(signal_path),
        "--layout-hint",
        layout_hint or "auto",
    ]
    if calibrated_signal_path is not None:
        command.extend(["--worker-calibrated-signal", str(calibrated_signal_path)])
    start = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        signal_path.unlink(missing_ok=True)
        if calibrated_signal_path is not None:
            calibrated_signal_path.unlink(missing_ok=True)
        record = _failure_record(
            image_path,
            mode=mode,
            source_sha256=source_sha256,
            layout_hint=layout_hint,
            status="timeout",
            elapsed_seconds=time.monotonic() - start,
            error=f"Timed out after {timeout_seconds} seconds",
        )
        _write_json(record_path, record)
        return record

    if process.returncode == 0 and record_path.exists():
        try:
            return json.loads(record_path.read_text())
        except json.JSONDecodeError:
            pass

    signal_path.unlink(missing_ok=True)
    if calibrated_signal_path is not None:
        calibrated_signal_path.unlink(missing_ok=True)
    error_tail = (process.stderr or process.stdout or "Worker produced no report")[-2000:]
    record = _failure_record(
        image_path,
        mode=mode,
        source_sha256=source_sha256,
        layout_hint=layout_hint,
        status="failed",
        elapsed_seconds=time.monotonic() - start,
        error=f"Worker exit code {process.returncode}: {error_tail.strip()}",
    )
    _write_json(record_path, record)
    return record


def _mode_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    timed_out = [record for record in records if record.get("status") == "timeout"]
    twelve_active = [
        record
        for record in successful
        if record.get("diagnostics", {}).get("nonzero_leads_count") == 12
    ]
    einthoven_scores = [
        float(record["diagnostics"]["einthoven_score"])
        for record in successful
        if isinstance(record.get("diagnostics", {}).get("einthoven_score"), (int, float))
    ]
    elapsed = [
        float(record["elapsed_seconds"])
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    total = len(records)
    success_rate = len(successful) / total if total else 0.0
    twelve_active_rate = len(twelve_active) / len(successful) if successful else 0.0
    median_einthoven = statistics.median(einthoven_scores) if einthoven_scores else None
    median_elapsed = statistics.median(elapsed) if elapsed else None
    acceptance = {
        "digitization_success_rate": success_rate >= 0.90,
        "twelve_active_leads_rate": twelve_active_rate >= 0.90,
        "median_einthoven_score": median_einthoven is not None and median_einthoven >= 0.70,
    }
    return {
        "total": total,
        "successful": len(successful),
        "failed": total - len(successful) - len(timed_out),
        "timed_out": len(timed_out),
        "with_warnings": sum(bool(record.get("has_warnings")) for record in successful),
        "success_rate": success_rate,
        "twelve_active_leads": len(twelve_active),
        "twelve_active_leads_rate": twelve_active_rate,
        "median_einthoven_score": median_einthoven,
        "median_elapsed_seconds": median_elapsed,
        "acceptance": acceptance,
        "acceptance_all_passed": all(acceptance.values()),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate operational quality metrics independently for each mode."""
    modes = sorted({str(record.get("mode", "unknown")) for record in records})
    variants = sorted({_record_variant(record) for record in records})
    return {
        "total_attempts": len(records),
        "modes": {
            mode: _mode_aggregate([record for record in records if record.get("mode") == mode])
            for mode in modes
        },
        "variants": {
            variant: {
                "modes": {
                    mode: _mode_aggregate(
                        [
                            record
                            for record in records
                            if record.get("mode") == mode and _record_variant(record) == variant
                        ]
                    )
                    for mode in modes
                }
            }
            for variant in variants
        },
    }


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("diagnostics", {})
    return {
        "source_image": record.get("source_image"),
        "mode": record.get("mode"),
        "capture_variant": _record_variant(record),
        "layout_hint": record.get("layout_hint"),
        "status": record.get("status"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "has_warnings": record.get("has_warnings"),
        "error": record.get("error"),
        "layout_name": diagnostics.get("layout_name"),
        "processing_mode": diagnostics.get("processing_mode"),
        "raw_lines_count": diagnostics.get("raw_lines_count"),
        "detected_leads_count": diagnostics.get("detected_leads_count"),
        "nonzero_leads_count": diagnostics.get("nonzero_leads_count"),
        "einthoven_score": diagnostics.get("einthoven_score"),
        "layout_cost": diagnostics.get("layout_cost"),
        "avg_pixel_per_mm": diagnostics.get("avg_pixel_per_mm"),
    }


def write_reports(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write aggregate JSON and flat per-attempt CSV reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "real_phone_evaluation.json"
    csv_path = output_dir / "real_phone_evaluation.csv"
    report = {
        "pipeline_version": REAL_PHOTO_EVALUATION_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "reference_level": "photos_only",
        "limitations": (
            "Operational digitization quality only; no matched reference signal was provided, "
            "so signal fidelity and diagnostic accuracy were not measured."
        ),
        "acceptance_thresholds": {
            "digitization_success_rate": 0.90,
            "twelve_active_leads_rate": 0.90,
            "median_einthoven_score": 0.70,
        },
        "aggregate": aggregate_records(records),
        "records": records,
    }
    _write_json(json_path, report)

    fieldnames = list(_csv_row(records[0]).keys()) if records else []
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(_csv_row(record) for record in records)
    return json_path, csv_path


def _print_summary(report_path: Path, records: list[dict[str, Any]]) -> None:
    print("\nReal phone photo evaluation")
    print(f"Report: {report_path}")
    for mode, metrics in aggregate_records(records)["modes"].items():
        einthoven = metrics["median_einthoven_score"]
        einthoven_text = "N/A" if einthoven is None else f"{einthoven:.3f}"
        print(
            f"  {mode}: success={metrics['successful']}/{metrics['total']} "
            f"({metrics['success_rate']:.1%}), "
            f"12-active={metrics['twelve_active_leads_rate']:.1%}, "
            f"median Einthoven={einthoven_text}, "
            f"timeouts={metrics['timed_out']}"
        )


def _worker_main(args: argparse.Namespace) -> None:
    image_path = Path(args.worker_image)
    record = run_worker(
        image_path,
        mode=args.worker_mode,
        layout_hint=None if args.layout_hint == "auto" else args.layout_hint,
        signal_path=Path(args.worker_signal),
        calibrated_signal_path=(
            Path(args.worker_calibrated_signal)
            if args.worker_calibrated_signal is not None
            else None
        ),
    )
    _write_json(Path(args.worker_output), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modes", nargs="+", choices=VALID_MODES, default=list(VALID_MODES))
    parser.add_argument("--layout-hint", default="auto")
    parser.add_argument("--use-metadata-layouts", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-image", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=VALID_MODES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-signal", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-calibrated-signal", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_image is not None:
        _worker_main(args)
        return

    image_paths = collect_photo_paths(args.image_dir, args.max_samples)
    metadata_layouts = load_layout_hints(args.metadata) if args.use_metadata_layouts else {}
    records: list[dict[str, Any]] = []
    for mode in args.modes:
        for index, image_path in enumerate(image_paths, start=1):
            layout_hint = (
                args.layout_hint
                if args.layout_hint != "auto"
                else metadata_layouts.get(_case_id(image_path))
            )
            source_sha256 = sha256_file(image_path)
            record_path = args.output_dir / "records" / mode / f"{image_path.stem}.json"
            signal_path = args.signal_dir / mode / f"{image_path.stem}.npy"
            print(
                f"[{mode}] {index}/{len(image_paths)} {image_path.name} "
                f"(layout={layout_hint or 'auto'})",
                flush=True,
            )
            if not args.overwrite and is_reusable_record(
                record_path,
                mode=mode,
                source_sha256=source_sha256,
                layout_hint=layout_hint,
            ):
                records.append(json.loads(record_path.read_text()))
                print("  reused current successful audit record", flush=True)
                continue
            record = run_isolated_attempt(
                image_path,
                mode=mode,
                layout_hint=layout_hint,
                signal_path=signal_path,
                record_path=record_path,
                timeout_seconds=args.timeout,
            )
            records.append(record)
            print(
                f"  {record['status']} in {record['elapsed_seconds']:.1f}s",
                flush=True,
            )

    json_path, _ = write_reports(records, args.output_dir)
    _print_summary(json_path, records)


if __name__ == "__main__":
    main()
