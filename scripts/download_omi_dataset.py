# Download the Chongqing ACS/OMI 12-lead ECG dataset from Figshare (CC0)
# Fetches the three archives, verifies MD5, and extracts into data/raw/omi-chongqing/

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR: Path = PROJECT_ROOT / "data" / "raw" / "omi-chongqing"

FIGSHARE_ARTICLE_ID: str = "29925314"
DATASET_DOI: str = "10.6084/m9.figshare.29925314"
DATASET_LICENSE: str = "CC0 1.0"

# Pinned file ids and checksums from Figshare article version 1. Pinning them
# means a silently republished version fails loudly instead of mixing data.
DATASET_FILES: list[dict[str, str | int]] = [
    {
        "name": "CSV.zip",
        "url": "https://ndownloader.figshare.com/files/62951134",
        "md5": "1cb46279c0e68e6512bd39214fc56528",
        "size": 296769,
    },
    {
        "name": "ECG_median_data.zip",
        "url": "https://ndownloader.figshare.com/files/62951137",
        "md5": "4da1650cdbfb9817aa66df58c66564e1",
        "size": 140304598,
    },
    {
        "name": "ECG_row_data.zip",
        "url": "https://ndownloader.figshare.com/files/62951395",
        "md5": "acea6ca86a2d0b937ecfe7d1df6d30a2",
        "size": 1277553270,
    },
]


def compute_md5(path: Path) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_size: int) -> None:
    """Stream a URL to disk, printing coarse progress."""
    print(f"[INFO] Downloading {destination.name} ({expected_size / 1e6:.0f} MB)...")

    def report(block_count: int, block_size: int, total_size: int) -> None:
        downloaded = block_count * block_size
        total = total_size if total_size > 0 else expected_size
        percent = min(100.0, 100.0 * downloaded / total)
        print(f"\r  {percent:5.1f}%  {downloaded / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()


def ensure_file(entry: dict, target_dir: Path, force: bool) -> Path:
    """Download one archive if needed and verify its checksum."""
    destination = target_dir / str(entry["name"])

    if destination.exists() and not force:
        print(f"[INFO] {destination.name} already present — verifying checksum")
    else:
        download_file(str(entry["url"]), destination, int(entry["size"]))

    actual_md5 = compute_md5(destination)
    if actual_md5 != entry["md5"]:
        print(
            f"[ERROR] Checksum mismatch for {destination.name}\n"
            f"  expected {entry['md5']}\n  actual   {actual_md5}\n"
            "  Delete the file and re-run, or check whether Figshare "
            "published a new dataset version."
        )
        sys.exit(1)
    print(f"[OK] {destination.name} checksum verified")
    return destination


def extract_archive(archive: Path, target_dir: Path) -> None:
    """Extract a zip archive into the target directory."""
    print(f"[INFO] Extracting {archive.name}...")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target_dir)


def main() -> None:
    """CLI entry point for downloading the OMI dataset."""
    parser = argparse.ArgumentParser(
        description="Download the Chongqing ACS/OMI ECG dataset from Figshare"
    )
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Fetch only the 0.3 MB label archive (no waveforms)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Verify archives without extracting"
    )
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] DOI {DATASET_DOI} — license {DATASET_LICENSE}")
    print(f"[INFO] Target: {args.target_dir}\n")

    entries = DATASET_FILES[:1] if args.csv_only else DATASET_FILES
    for entry in entries:
        archive = ensure_file(entry, args.target_dir, args.force)
        if not args.skip_extract:
            extract_archive(archive, args.target_dir)

    print("\n[DONE] Dataset ready.")
    print(
        "  Note: test.csv ships without labels. Test-set metrics come from the "
        "authors' online evaluation platform (http://39.105.59.221:8080/), which "
        "accepts binary predictions only."
    )


if __name__ == "__main__":
    main()
