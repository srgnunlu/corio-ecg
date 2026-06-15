"""Build a validated Level B manifest from a strict local CSV inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.matched_photo_builder import (  # noqa: E402
    build_matched_photo_manifest,
    load_collection_inventory,
)
from src.evaluation.matched_photo_manifest import (  # noqa: E402
    DEFAULT_MATCHED_PHOTO_MANIFEST_CONFIG,
    load_matched_photo_manifest_config,
)

DEFAULT_DATASET_ROOT = Path("data/level-b-matched")
DEFAULT_INVENTORY = DEFAULT_DATASET_ROOT / "collection.csv"
DEFAULT_OUTPUT = DEFAULT_DATASET_ROOT / "manifest.json"


def write_manifest(
    manifest: dict[str, object],
    output_path: Path,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Write a validated local manifest without accidental overwrite."""
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--config", type=Path, default=DEFAULT_MATCHED_PHOTO_MANIFEST_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    config = load_matched_photo_manifest_config(args.config)
    manifest = build_matched_photo_manifest(
        load_collection_inventory(args.inventory),
        args.dataset_root,
        config,
    )
    write_manifest(manifest, args.output, allow_overwrite=args.allow_overwrite)
    print(f"Wrote validated manifest: {args.output}")
    print(f"Records: {len(manifest['records'])}")


if __name__ == "__main__":
    main()
