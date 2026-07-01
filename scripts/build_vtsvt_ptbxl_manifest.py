"""Build a PTB-XL development manifest for VT/SVT criteria audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.vtsvt_ptbxl_manifest import (  # noqa: E402
    VTSVTPTBXLManifestConfig,
    build_ptbxl_vtsvt_manifest,
    load_ptbxl_metadata,
    select_ptbxl_vtsvt_records,
    write_manifest,
)

DEFAULT_DATA_DIR = Path("data/raw/ptb-xl")
DEFAULT_OUTPUT = Path("data/splits/vtsvt_ptbxl_audit_manifest_v1.yaml")

__all__ = [
    "VTSVTPTBXLManifestConfig",
    "build_ptbxl_vtsvt_manifest",
    "load_ptbxl_metadata",
    "select_ptbxl_vtsvt_records",
    "write_manifest",
]


def main() -> None:
    """Parse CLI args and write the generated PTB-XL VTSVT manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--strat-fold", type=int, default=10)
    parser.add_argument("--per-category-limit", type=int, default=12)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    config = VTSVTPTBXLManifestConfig(
        seed=args.seed,
        strat_fold=args.strat_fold,
        per_category_limit=args.per_category_limit,
    )
    metadata = load_ptbxl_metadata(args.data_dir)
    manifest = build_ptbxl_vtsvt_manifest(metadata, args.data_dir, config)
    write_manifest(manifest, args.output, allow_overwrite=args.allow_overwrite)

    summary = manifest["summary"]
    print(f"Wrote {args.output}")
    print(f"Selected records: {summary['selected_records']}")
    print(f"Categories: {summary['category_counts']}")


if __name__ == "__main__":
    main()
