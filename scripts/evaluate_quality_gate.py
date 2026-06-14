"""Benchmark the interpretable digitization quality gate on PMcardio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.quality_gate import (  # noqa: E402
    QualityGateConfig,
    classify_fidelity_target,
    classify_quality,
    evaluate_quality_gate,
    load_quality_gate_config,
)
from src.training.quality_gate_config import DEFAULT_QUALITY_GATE_CONFIG_PATH  # noqa: E402

DEFAULT_INPUT = Path("results/pmcardio-reference/pmcardio_reference_fidelity.json")
DEFAULT_OUTPUT_DIR = Path("results/quality-gate")


class EvaluationStage(StrEnum):
    """Allowed benchmark evaluation stages."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"
    EXTERNAL = "external"


def calculate_file_sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hash(path: Path, expected_sha256: str | None = None) -> str:
    """Return the source digest and reject an unexpected source artifact."""
    actual_sha256 = calculate_file_sha256(path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def build_report(
    source_report: dict[str, Any],
    config: QualityGateConfig | None = None,
    evaluation_stage: EvaluationStage = EvaluationStage.DEVELOPMENT,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a quality-gate benchmark report from matched-reference records."""
    resolved_config = config or load_quality_gate_config()
    records = source_report["records"]
    evaluated_records = []
    for record in records:
        decision = classify_quality(record, resolved_config)
        evaluated_records.append(
            {
                "category": record.get("category"),
                "image_id": record.get("image_id"),
                "ecg_id": record.get("ecg_id"),
                "target": classify_fidelity_target(record, resolved_config).value,
                "prediction": decision.outcome.value,
                "reasons": list(decision.reasons),
            }
        )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "gate_version": resolved_config.version,
        "quality_gate_config": resolved_config.to_dict(),
        "evaluation_stage": evaluation_stage.value,
        "source_sha256": source_sha256,
        "development_status": (
            "Thresholds were developed on this same 70-image PMcardio subset. "
            "Results are exploratory and require external matched-reference validation."
        ),
        "source_manifest": source_report.get("selection_manifest"),
        "target_definition": {
            "reject": "correlation < 0.60, RMSE > 0.20 mV, SNR < 0 dB, or failure",
            "warn": "correlation < 0.80, RMSE > 0.15 mV, or SNR < 3 dB",
            "accept": "all reject and warn thresholds passed",
        },
        "gate_definition": {
            "reject": (
                "failure, missing diagnostics, active leads < 10, detected labels < 4, "
                "or at least two severe independent risk flags"
            ),
            "severe_risk_flags": (
                "Einthoven < 0.60, layout cost > 0.60, detected labels < 6, "
                "pixel density < 8.8 px/mm, or active leads < 12"
            ),
            "warn": (
                "detected labels < 8, Einthoven < 0.90, layout cost > 0.60, "
                "pixel density < 8.8 px/mm, or active leads < 12"
            ),
            "accept": "all reject and warn thresholds passed",
        },
        "limitations": (
            "The gate predicts reconstruction fidelity risk, not clinical diagnostic "
            "accuracy. Zero false accepts on this development subset does not establish "
            "safety on unseen ECG photos."
        ),
        "aggregate": evaluate_quality_gate(records, resolved_config),
        "records": evaluated_records,
    }


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    allow_overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write JSON and CSV quality-gate reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_gate_benchmark.json"
    csv_path = output_dir / "quality_gate_benchmark.csv"
    existing_paths = [path for path in (json_path, csv_path) if path.exists()]
    if existing_paths and not allow_overwrite:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            f"Refusing to overwrite existing quality-gate artifact(s): {names}. "
            "Use --allow-overwrite only for an intentional replacement."
        )
    json_path.write_text(json.dumps(report, indent=2))
    rows = report["records"]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_QUALITY_GATE_CONFIG_PATH)
    parser.add_argument(
        "--evaluation-stage",
        choices=[stage.value for stage in EvaluationStage],
        default=EvaluationStage.DEVELOPMENT.value,
    )
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    source_sha256 = validate_source_hash(args.input, args.expected_input_sha256)
    source_report = json.loads(args.input.read_text())
    config = load_quality_gate_config(args.config)
    report = build_report(
        source_report,
        config=config,
        evaluation_stage=EvaluationStage(args.evaluation_stage),
        source_sha256=source_sha256,
    )
    json_path, csv_path = write_report(
        report,
        args.output_dir,
        allow_overwrite=args.allow_overwrite,
    )
    aggregate = report["aggregate"]
    print(f"Wrote {json_path} and {csv_path}")
    print(
        f"False accepts: {aggregate['false_accepts']} "
        f"({aggregate['false_accept_rate']:.1%})"
    )
    print(
        f"False rejects: {aggregate['false_rejects']} "
        f"({aggregate['false_reject_rate']:.1%})"
    )
    print(
        f"Missed rejects: {aggregate['missed_rejects']} "
        f"(reject recall {aggregate['reject_recall']:.1%})"
    )


if __name__ == "__main__":
    main()
