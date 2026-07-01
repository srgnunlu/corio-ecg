"""CLI wrapper for deterministic Brugada/Vereckei VT/SVT criteria audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.vtsvt_audit import (  # noqa: E402
    aggregate_records,
    build_audit_report,
    evaluate_manifest,
)
from src.evaluation.vtsvt_audit_io import (  # noqa: E402
    load_manifest,
    load_signal_record,
    write_report,
)
from src.evaluation.vtsvt_audit_schema import LoadedSignal  # noqa: E402

DEFAULT_MANIFEST = Path("configs/vtsvt_audit_manifest_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("results/vtsvt")

__all__ = [
    "LoadedSignal",
    "aggregate_records",
    "build_audit_report",
    "evaluate_manifest",
    "load_manifest",
    "load_signal_record",
    "write_report",
]


def main() -> None:
    """Parse CLI arguments and run the VTSVT audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    report = evaluate_manifest(manifest, args.manifest)
    json_path, csv_path = write_report(
        report,
        args.output_dir,
        allow_overwrite=args.allow_overwrite,
    )
    aggregate = report["aggregate"]
    print(f"Wrote {json_path} and {csv_path}")
    print(f"Success: {aggregate['successful']}/{aggregate['total']}")
    print(f"VT supported: {aggregate['supports_vt']}")


if __name__ == "__main__":
    main()
