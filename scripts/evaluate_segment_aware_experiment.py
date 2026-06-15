"""Apply the research-only segment-aware decision contract to a drift report."""

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
from src.evaluation.segment_aware_experiment import (  # noqa: E402
    DEFAULT_SEGMENT_AWARE_EXPERIMENT_CONFIG,
    evaluate_segment_aware_experiment,
    load_segment_aware_experiment_config,
)

DEFAULT_INPUT = Path("results/pmcardio-reference/pmcardio_diagnosis_drift.json")
DEFAULT_OUTPUT_DIR = Path("results/digitization-experiments/segment-aware-v1")


def build_report(comparison: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    """Attach source provenance to a segment-aware comparison."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_sha256,
        **comparison,
    }


def _category_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "role": result["role"],
            **result["source_counts"],
            **{f"baseline_{key}": value for key, value in result["baseline"].items()},
            **{f"candidate_{key}": value for key, value in result["candidate"].items()},
            **{f"delta_{key}": value for key, value in result["deltas"].items()},
        }
        for category, result in report["categories"].items()
    ]


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    allow_overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write audit-safe JSON and category-level CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "segment_aware_experiment.json"
    csv_path = output_dir / "segment_aware_experiment.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not allow_overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifact(s): {names}"
        )
    json_path.write_text(json.dumps(report, indent=2))
    rows = _category_rows(report)
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_SEGMENT_AWARE_EXPERIMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    comparison = evaluate_segment_aware_experiment(
        json.loads(args.input.read_text()),
        config=load_segment_aware_experiment_config(args.config),
    )
    report = build_report(comparison, source_sha256=calculate_file_sha256(args.input))
    json_path, csv_path = write_report(
        report,
        args.output_dir,
        allow_overwrite=args.allow_overwrite,
    )
    print(f"Wrote {json_path} and {csv_path}")
    print(f"Research candidate passed: {report['research_candidate']['passed']}")
    print(f"Production status: {report['production_status']}")
    if not report["research_candidate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
