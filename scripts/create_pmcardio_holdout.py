"""Create the pre-registered PMcardio internal holdout manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pmcardio_holdout import (  # noqa: E402
    DEFAULT_PMCARDIO_HOLDOUT_CONFIG,
    build_pmcardio_holdout_manifest,
    load_pmcardio_holdout_config,
)

DEFAULT_METADATA = Path("data/reference/pmcardio/metadata.csv")
DEFAULT_DEVELOPMENT_MANIFEST = Path("data/reference/pmcardio/selection_manifest.json")
DEFAULT_OUTPUT = Path("results/pmcardio-holdout/pmcardio_holdout_v1.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    manifest: dict[str, object],
    output_path: Path,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Write a pre-registration artifact without accidental replacement."""
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite holdout manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PMCARDIO_HOLDOUT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_pmcardio_holdout_manifest(
        pd.read_csv(args.metadata),
        json.loads(args.development_manifest.read_text()),
        load_pmcardio_holdout_config(args.config),
        metadata_sha256=_sha256(args.metadata),
    )
    write_manifest(manifest, args.output, allow_overwrite=args.allow_overwrite)
    print(f"Wrote {args.output}")
    print(f"Manifest ID: {manifest['manifest_id']}")
    print(f"Summary: {manifest['summary']}")


if __name__ == "__main__":
    main()
