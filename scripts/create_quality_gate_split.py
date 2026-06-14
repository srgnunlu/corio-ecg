"""Create a deterministic ECG-grouped quality-gate split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.grouped_split import (  # noqa: E402
    DEFAULT_GROUPED_SPLIT_CONFIG_PATH,
    GroupedSplitConfig,
    create_grouped_split_manifest,
    load_grouped_split_config,
)

DEFAULT_INPUT = Path("results/pmcardio-reference/pmcardio_reference_fidelity.json")
DEFAULT_OUTPUT = Path("results/quality-gate/quality_gate_split_v1.json")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_split_artifact(
    source_path: Path,
    config: GroupedSplitConfig,
) -> dict[str, Any]:
    """Build an auditable split artifact from a matched-reference report."""
    source_bytes = source_path.read_bytes()
    source_report = json.loads(source_bytes)
    manifest = create_grouped_split_manifest(source_report["records"], config)
    artifact = {
        "source_path": str(source_path),
        "source_sha256": _sha256_bytes(source_bytes),
        **manifest,
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    artifact["manifest_id"] = _sha256_bytes(canonical)
    return artifact


def write_split_artifact(
    artifact: dict[str, Any],
    output_path: Path,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Write a split artifact while protecting existing manifests."""
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing split manifest: {output_path}. "
            "Use --allow-overwrite only for intentional regeneration."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_GROUPED_SPLIT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    config = load_grouped_split_config(args.config)
    artifact = build_split_artifact(args.input, config)
    write_split_artifact(artifact, args.output, allow_overwrite=args.allow_overwrite)
    summary = artifact["summary"]
    print(f"Wrote {args.output}")
    print(f"Manifest ID: {artifact['manifest_id']}")
    print(f"ECG groups: {summary['group_counts']}")
    print(f"Records: {summary['record_counts']}")


if __name__ == "__main__":
    main()
