"""Download a small matched PMcardio image/reference subset from remote ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd

PMCARDIO_RECORD_URL = "https://zenodo.org/records/13617673"
PMCARDIO_ARCHIVE_URL = (
    "https://zenodo.org/api/records/13617673/files/"
    "pmcardio_ecg_image_database.zip/content"
)
PMCARDIO_DOI = "10.5281/zenodo.13617673"
PMCARDIO_LICENSE = "GPL-3.0-or-later"
DEFAULT_OUTPUT_DIR = Path("data/reference/pmcardio")
DEFAULT_IMAGE_IDS = (4, 10, 14)
DEFAULT_SELECTION_SEED = 20260613
PHYSICAL_CATEGORIES = (
    "photos_bents",
    "photos_crumbles",
    "photos_doogee",
    "photos_iphone",
    "photos_samsung",
    "photos_scans",
    "photos_screens",
)
REFERENCE_MEMBERS = (
    "final_data/data/leads.npz",
    "final_data/data/rhythms.npz",
    "final_data/metadata.csv",
)


class RangeHTTPFile(io.RawIOBase):
    """Seekable read-only HTTP file backed by Range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.position = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed trusted URL
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before start of remote file")
        self.position = position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.size - 1 if size < 0 else min(self.position + size - 1, self.size - 1)
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed trusted URL
            data = response.read()
        self.position += len(data)
        return data


def _category_from_path(relative_path: str) -> str:
    return relative_path.split("/", maxsplit=1)[0]


def select_reference_rows(
    metadata: pd.DataFrame,
    *,
    image_ids: list[int] | tuple[int, ...],
    categories: list[str] | tuple[str, ...],
    layout: str,
) -> pd.DataFrame:
    """Select and validate one page for every requested ID/category pair."""
    selected = metadata.copy()
    selected["category"] = selected["Image relative path"].map(_category_from_path)
    selected = selected[
        selected["Image ID"].isin(image_ids)
        & selected["category"].isin(categories)
        & (selected["ECG format"] == layout)
        & (selected["Image page"] == 0)
    ].copy()
    expected_pairs = {(image_id, category) for image_id in image_ids for category in categories}
    actual_pairs = set(zip(selected["Image ID"], selected["category"], strict=False))
    missing = sorted(expected_pairs - actual_pairs)
    if missing:
        raise ValueError(f"Missing requested PMcardio variants: {missing}")
    return selected.sort_values(["Image ID", "category"]).reset_index(drop=True)


def select_balanced_image_ids(
    metadata: pd.DataFrame,
    *,
    categories: list[str] | tuple[str, ...],
    layout: str,
    count: int,
    seed: int,
    required_ids: list[int] | tuple[int, ...],
) -> list[int]:
    """Select deterministic image IDs available in every requested category."""
    if count <= 0:
        raise ValueError("count must be positive")

    candidates = metadata.copy()
    candidates["category"] = candidates["Image relative path"].map(_category_from_path)
    candidates = candidates[
        candidates["category"].isin(categories)
        & (candidates["ECG format"] == layout)
        & (candidates["Image page"] == 0)
    ]
    ids_by_category = [
        set(int(value) for value in candidates[candidates["category"] == category]["Image ID"])
        for category in categories
    ]
    common_ids = set.intersection(*ids_by_category) if ids_by_category else set()
    required = sorted(set(int(value) for value in required_ids))
    missing_required = sorted(set(required) - common_ids)
    if missing_required:
        raise ValueError(f"Required image IDs are not common to every category: {missing_required}")
    if len(common_ids) < count:
        raise ValueError(
            f"Requested {count} balanced image IDs, but only {len(common_ids)} "
            "common image IDs are available"
        )
    if len(required) > count:
        raise ValueError("required_ids contains more IDs than the requested count")

    remaining = sorted(common_ids - set(required))
    random.Random(seed).shuffle(remaining)
    return sorted(required + remaining[: count - len(required)])


def build_selection_manifest(
    selected: pd.DataFrame,
    *,
    selection_seed: int | None,
) -> dict[str, object]:
    """Build a stable, content-addressed manifest for a selected subset."""
    ordered = selected.sort_values(["Image ID", "category", "Image relative path"])
    identity_columns = [
        "Image ID",
        "ECG ID",
        "Image relative path",
        "Image page",
        "ECG format",
        "category",
    ]
    identity_records = ordered[identity_columns].to_dict(orient="records")
    canonical = json.dumps(identity_records, sort_keys=True, separators=(",", ":"))
    manifest_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    counts = ordered.groupby("category").size().sort_index()
    return {
        "manifest_version": 1,
        "manifest_id": manifest_id,
        "selection_seed": selection_seed,
        "selection": {
            "image_ids": sorted(int(value) for value in ordered["Image ID"].unique()),
            "categories": sorted(str(value) for value in ordered["category"].unique()),
            "layout": sorted(str(value) for value in ordered["ECG format"].unique()),
            "images": len(ordered),
            "images_per_category": {
                str(category): int(count) for category, count in counts.items()
            },
        },
        "records": identity_records,
    }


def reference_key_for_row(row: pd.Series) -> str:
    """Return the leads.npz key that corresponds to a metadata row."""
    category = _category_from_path(str(row["Image relative path"]))
    ecg_id = str(row["ECG ID"])
    if category.startswith("digital_data_"):
        return f"{category.removeprefix('digital_data_')}_{ecg_id}"
    return ecg_id


def _copy_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _read_metadata(archive: zipfile.ZipFile) -> pd.DataFrame:
    with archive.open("final_data/metadata.csv") as source:
        return pd.read_csv(source)


def _write_source_metadata(
    output_dir: Path,
    selected: pd.DataFrame,
    *,
    selection_seed: int | None,
) -> None:
    manifest = build_selection_manifest(selected, selection_seed=selection_seed)
    payload = {
        "name": "PMcardio ECG Image Database (PM-ECG-ID)",
        "source_url": PMCARDIO_RECORD_URL,
        "doi": PMCARDIO_DOI,
        "license": PMCARDIO_LICENSE,
        "manifest_id": manifest["manifest_id"],
        "selection": manifest["selection"],
    }
    (output_dir / "source.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))


def download_subset(
    output_dir: Path,
    *,
    image_ids: list[int] | tuple[int, ...],
    categories: list[str] | tuple[str, ...],
    layout: str,
    balanced_count: int | None = None,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    required_image_ids: list[int] | tuple[int, ...] = (),
) -> pd.DataFrame:
    """Download selected images and all compact reference arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_file: BinaryIO = RangeHTTPFile(PMCARDIO_ARCHIVE_URL)
    with zipfile.ZipFile(remote_file) as archive:
        metadata = _read_metadata(archive)
        if balanced_count is not None:
            image_ids = select_balanced_image_ids(
                metadata,
                categories=categories,
                layout=layout,
                count=balanced_count,
                seed=selection_seed,
                required_ids=required_image_ids,
            )
        selected = select_reference_rows(
            metadata,
            image_ids=image_ids,
            categories=categories,
            layout=layout,
        )
        for member in REFERENCE_MEMBERS:
            destination = output_dir / Path(member).name
            if not destination.exists():
                print(f"Downloading reference: {member}", flush=True)
                _copy_member(archive, member, destination)
        for _, row in selected.iterrows():
            relative_path = str(row["Image relative path"])
            member = f"final_data/visual_data/{relative_path}"
            destination = output_dir / "images" / relative_path
            if not destination.exists():
                print(f"Downloading image: {relative_path}", flush=True)
                _copy_member(archive, member, destination)

    selected["reference_key"] = selected.apply(reference_key_for_row, axis=1)
    selected.to_csv(output_dir / "subset_metadata.csv", index=False)
    _write_source_metadata(
        output_dir,
        selected,
        selection_seed=selection_seed if balanced_count is not None else None,
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-ids", nargs="+", type=int, default=list(DEFAULT_IMAGE_IDS))
    parser.add_argument("--categories", nargs="+", default=list(PHYSICAL_CATEGORIES))
    parser.add_argument("--layout", default="3x4+1R")
    parser.add_argument(
        "--balanced-count",
        type=int,
        default=None,
        help="Deterministically select this many IDs available in every category",
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--required-image-ids",
        nargs="*",
        type=int,
        default=list(DEFAULT_IMAGE_IDS),
        help="IDs that a balanced selection must preserve",
    )
    args = parser.parse_args()

    selected = download_subset(
        args.output_dir,
        image_ids=args.image_ids,
        categories=args.categories,
        layout=args.layout,
        balanced_count=args.balanced_count,
        selection_seed=args.selection_seed,
        required_image_ids=args.required_image_ids,
    )
    print(f"Downloaded {len(selected)} matched PMcardio reference images.")


if __name__ == "__main__":
    main()
