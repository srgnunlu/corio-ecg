# Manifest, signal, and artifact IO for VT/SVT criteria audits.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

from src.evaluation.vtsvt_audit_schema import CSV_FIELDS, LoadedSignal
from src.utils.wfdb_helpers import read_ecg_signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_BASENAME = "vtsvt_criteria_audit_v1"


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a YAML audit manifest."""
    manifest = yaml.safe_load(path.read_text()) or {}
    if not isinstance(manifest.get("records"), list):
        raise ValueError("VTSVT audit manifest must contain a records list")
    return manifest


def load_signal_record(
    record: dict[str, Any],
    manifest_dir: Path,
    default_sample_rate: int,
) -> LoadedSignal:
    """Load a manifest record from `.npy` or WFDB input."""
    signal_path = record.get("signal_path") or record.get("record_path")
    if not signal_path:
        raise ValueError("manifest record requires signal_path or record_path")

    resolved_path = _resolve_path(Path(str(signal_path)), manifest_dir)
    sample_rate = int(record.get("sample_rate") or default_sample_rate)
    if resolved_path.suffix == ".npy":
        signal = _as_12_lead_signal(np.load(resolved_path))
    else:
        signal, metadata = read_ecg_signal(str(resolved_path))
        sample_rate = int(metadata.get("sample_rate", sample_rate))

    rhythm_strip = None
    if rhythm_path := record.get("rhythm_strip_path"):
        resolved_rhythm = _resolve_path(Path(str(rhythm_path)), manifest_dir)
        rhythm_strip = _as_12_lead_signal(np.load(resolved_rhythm))

    return LoadedSignal(
        signal=signal.astype(np.float32),
        rhythm_strip=rhythm_strip,
        sample_rate=sample_rate,
        source_path=str(resolved_path),
    )


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    allow_overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write JSON and CSV audit artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{REPORT_BASENAME}.json"
    csv_path = output_dir / f"{REPORT_BASENAME}.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not allow_overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing VTSVT audit artifact(s): {names}. "
            "Use --allow-overwrite for an intentional replacement."
        )

    json_path.write_text(json.dumps(report, indent=2))
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in report["records"]:
            writer.writerow({field: _csv_value(record.get(field)) for field in CSV_FIELDS})
    return json_path, csv_path


def _as_12_lead_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("signal must have shape (12, samples) or (samples, 12)")
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    raise ValueError("signal must have shape (12, samples) or (samples, 12)")


def _resolve_path(path: Path, manifest_dir: Path) -> Path:
    if path.is_absolute():
        return path
    project_relative = PROJECT_ROOT / path
    if _path_exists(project_relative):
        return project_relative
    return manifest_dir / path


def _path_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".hea").exists()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value
