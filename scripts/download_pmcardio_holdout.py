"""Download only images locked in the pre-registered PMcardio holdout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_pmcardio_reference_subset import (  # noqa: E402
    PMCARDIO_ARCHIVE_URL,
    PMCARDIO_DOI,
    PMCARDIO_LICENSE,
    PMCARDIO_RECORD_URL,
    RangeHTTPFile,
    _copy_member,
    reference_key_for_row,
)

DEFAULT_MANIFEST = Path("results/pmcardio-holdout/pmcardio_holdout_v1.json")
DEFAULT_SOURCE_DIR = Path("data/reference/pmcardio")
DEFAULT_OUTPUT_DIR = Path("data/reference/pmcardio-holdout")
REFERENCE_FILES = ("leads.npz", "rhythms.npz", "metadata.csv")


def select_manifest_metadata(
    metadata: pd.DataFrame,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Select exactly the rows locked by the holdout manifest."""
    paths = [str(record["relative_path"]) for record in manifest["records"]]
    if len(paths) != len(set(paths)):
        raise ValueError("holdout manifest contains duplicate image paths")
    selected = metadata[metadata["Image relative path"].astype(str).isin(paths)].copy()
    actual = set(selected["Image relative path"].astype(str))
    missing = sorted(set(paths) - actual)
    unexpected_count = len(selected) - len(paths)
    if missing or unexpected_count:
        raise ValueError(
            f"metadata does not match holdout manifest: missing={missing}, "
            f"unexpected_count={unexpected_count}"
        )
    split_by_path = {
        str(record["relative_path"]): str(record["split"])
        for record in manifest["records"]
    }
    selected["category"] = selected["Image relative path"].map(
        lambda path: str(path).split("/", maxsplit=1)[0]
    )
    selected["split"] = selected["Image relative path"].map(split_by_path)
    selected["reference_key"] = selected.apply(reference_key_for_row, axis=1)
    return selected.sort_values(["split", "ECG ID", "category"]).reset_index(drop=True)


def _copy_reference_files(source_dir: Path, output_dir: Path) -> None:
    for filename in REFERENCE_FILES:
        source = source_dir / filename
        destination = output_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing local PMcardio reference file: {source}")
        if not destination.exists():
            shutil.copy2(source, destination)


def download_holdout(
    manifest_path: Path,
    source_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Download manifest images and prepare an isolated evaluation directory."""
    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_reference_files(source_dir, output_dir)
    metadata = pd.read_csv(source_dir / "metadata.csv")
    selected = select_manifest_metadata(metadata, manifest)

    remote_file = RangeHTTPFile(PMCARDIO_ARCHIVE_URL)
    with zipfile.ZipFile(remote_file) as archive:
        for relative_path in selected["Image relative path"].astype(str):
            destination = output_dir / "images" / relative_path
            if destination.exists():
                continue
            print(f"Downloading image: {relative_path}", flush=True)
            _copy_member(
                archive,
                f"final_data/visual_data/{relative_path}",
                destination,
            )

    selected.to_csv(output_dir / "subset_metadata.csv", index=False, lineterminator="\n")
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    source = {
        "name": "PMcardio ECG Image Database (PM-ECG-ID)",
        "source_url": PMCARDIO_RECORD_URL,
        "doi": PMCARDIO_DOI,
        "license": PMCARDIO_LICENSE,
        "holdout_manifest_id": manifest["manifest_id"],
        "evidence_status": manifest["evidence_status"],
    }
    (output_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    selected = download_holdout(args.manifest, args.source_dir, args.output_dir)
    print(f"Prepared {len(selected)} holdout images in {args.output_dir}")


if __name__ == "__main__":
    main()
