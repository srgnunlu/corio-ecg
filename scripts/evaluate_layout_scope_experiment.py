"""Evaluate the conservative layout-scope policy on PMcardio tune evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.layout_scope_experiment import (  # noqa: E402
    DEFAULT_LAYOUT_SCOPE_CONFIG,
    evaluate_layout_scope,
    load_layout_scope_config,
)

DEFAULT_QUALITY_REPORT = Path(
    "results/pmcardio-holdout/tune/quality-gate/quality_gate_benchmark.json"
)
DEFAULT_FIDELITY_REPORT = Path(
    "results/pmcardio-holdout/tune/pmcardio_reference_fidelity.json"
)
DEFAULT_OUTPUT_DIR = Path("results/pmcardio-holdout/tune/layout-scope")


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write aggregate JSON and reviewable record CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "layout_scope_experiment.json"
    csv_path = output_dir / "layout_scope_experiment.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    rows = report["records"]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--fidelity-report", type=Path, default=DEFAULT_FIDELITY_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_LAYOUT_SCOPE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = evaluate_layout_scope(
        json.loads(args.quality_report.read_text()),
        json.loads(args.fidelity_report.read_text()),
        load_layout_scope_config(args.config),
    )
    json_path, csv_path = write_report(report, args.output_dir)
    print(f"Wrote {json_path} and {csv_path}")
    print(f"Aggregate: {report['aggregate']}")


if __name__ == "__main__":
    main()
