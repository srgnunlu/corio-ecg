"""Compare a candidate digitization report against the frozen Phase 2 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_quality_gate import calculate_file_sha256  # noqa: E402
from src.evaluation.digitization_regression import (  # noqa: E402
    compare_digitization_reports,
    load_digitization_regression_config,
)
from src.evaluation.digitization_regression_config import (  # noqa: E402
    DEFAULT_DIGITIZATION_REGRESSION_CONFIG,
)

DEFAULT_BASELINE = Path("results/pmcardio-reference/pmcardio_reference_fidelity.json")
DEFAULT_OUTPUT_DIR = Path("results/digitization-experiments")


def build_report(
    comparison: dict[str, Any],
    *,
    experiment_name: str,
    baseline_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    """Attach experiment provenance to a regression comparison."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_name": experiment_name,
        "baseline_sha256": baseline_sha256,
        "candidate_sha256": candidate_sha256,
        **comparison,
    }


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    gate = record.get("quality_gate", {})
    return {
        **{key: value for key, value in record.items() if key != "quality_gate"},
        "baseline_quality_gate": gate.get("baseline"),
        "candidate_quality_gate": gate.get("candidate"),
    }


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    allow_overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write audit-safe JSON and record-level CSV experiment artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "digitization_regression.json"
    csv_path = output_dir / "digitization_regression.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not allow_overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifact(s): {names}"
        )
    json_path.write_text(json.dumps(report, indent=2))
    rows = [_flatten_record(record) for record in report["records"]]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0]) if rows else [],
            lineterminator="\n",
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--config", type=Path, default=DEFAULT_DIGITIZATION_REGRESSION_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    comparison = compare_digitization_reports(
        baseline,
        candidate,
        config=load_digitization_regression_config(args.config),
    )
    report = build_report(
        comparison,
        experiment_name=args.experiment_name,
        baseline_sha256=calculate_file_sha256(args.baseline),
        candidate_sha256=calculate_file_sha256(args.candidate),
    )
    json_path, csv_path = write_report(
        report,
        args.output_dir,
        allow_overwrite=args.allow_overwrite,
    )
    print(f"Wrote {json_path} and {csv_path}")
    print(f"Promotion passed: {report['promotion']['passed']}")
    if not report["promotion"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
