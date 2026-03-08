# Download PTB-XL dataset from Kaggle (fast) or PhysioNet (fallback)
# Fetches the archive, extracts it, and organizes files into data/raw/ptb-xl/

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Resolve project root relative to this script's location
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Kaggle dataset slug
KAGGLE_DATASET: str = "khyeh0719/ptb-xl-dataset"

# PhysioNet fallback URL
PHYSIONET_URL: str = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
PHYSIONET_ZIP_FILENAME: str = "ptb-xl-1.0.3.zip"
PHYSIONET_NESTED_DIR: str = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def run_command(command: list[str], description: str) -> None:
    """Run a shell command and exit on failure."""
    print(f"[INFO] Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed:")
        print(result.stderr)
        sys.exit(1)


def _flatten_nested_dir(data_directory: Path) -> None:
    """Move files from any single nested directory up to data_directory."""
    subdirs = [d for d in data_directory.iterdir() if d.is_dir()]
    # If there's exactly one subdirectory and it contains the dataset files
    for subdir in subdirs:
        csv_files = list(subdir.glob("*.csv"))
        if csv_files:
            print(f"[INFO] Flattening nested directory: {subdir.name}")
            for item in subdir.iterdir():
                destination = data_directory / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(destination))
            subdir.rmdir()
            return


def download_from_kaggle(data_directory: Path) -> bool:
    """Download PTB-XL from Kaggle API. Returns True on success."""
    # Check if kaggle credentials are available
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_creds = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ

    if not kaggle_json.exists() and not has_env_creds:
        print("[SKIP] No Kaggle credentials found")
        print("       Set KAGGLE_USERNAME + KAGGLE_KEY env vars, or place kaggle.json")
        return False

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[INFO] Installing kaggle package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "kaggle"],
            check=True,
        )

    print(f"[INFO] Downloading PTB-XL from Kaggle ({KAGGLE_DATASET})...")
    # Use kaggle CLI directly (not python -m kaggle which fails on some installs)
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(data_directory),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[WARN] Kaggle download failed: {result.stderr.strip()}")
        return False

    print("[OK] Kaggle download complete")
    _flatten_nested_dir(data_directory)
    return True


def download_from_physionet(data_directory: Path) -> None:
    """Download PTB-XL from PhysioNet (slower fallback)."""
    zip_file_path = data_directory / PHYSIONET_ZIP_FILENAME

    print(f"[INFO] Downloading PTB-XL from PhysioNet (~3 GB, may be slow)...")
    print(f"[INFO] Source: {PHYSIONET_URL}")

    run_command(
        ["curl", "-L", "-o", str(zip_file_path), "--progress-bar", PHYSIONET_URL],
        description="Download",
    )
    print("[OK] Download complete.")

    print("[INFO] Extracting archive...")
    run_command(
        ["unzip", "-q", "-o", str(zip_file_path), "-d", str(data_directory)],
        description="Extraction",
    )
    print("[OK] Extraction complete.")

    # Flatten nested PhysioNet directory structure
    nested_directory = data_directory / PHYSIONET_NESTED_DIR
    if nested_directory.exists():
        print(f"[INFO] Moving files from nested directory...")
        for item in nested_directory.iterdir():
            destination = data_directory / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
        nested_directory.rmdir()

    # Clean up zip
    if zip_file_path.exists():
        zip_file_path.unlink()
        print("[OK] Zip file removed.")


def download_ptbxl_dataset(source: str = "auto") -> None:
    """Download and extract the PTB-XL dataset.

    Args:
        source: Download source — 'kaggle', 'physionet', or 'auto' (try kaggle first).
    """
    data_directory = PROJECT_ROOT / "data" / "raw" / "ptb-xl"
    download_marker = data_directory / ".download_complete"

    if download_marker.exists():
        print(f"[OK] PTB-XL dataset already downloaded: {data_directory}")
        return

    data_directory.mkdir(parents=True, exist_ok=True)

    success = False

    if source in ("kaggle", "auto"):
        success = download_from_kaggle(data_directory)

    if not success and source in ("physionet", "auto"):
        if source == "auto":
            print("[INFO] Falling back to PhysioNet...")
        download_from_physionet(data_directory)
        success = True

    if not success:
        print("[ERROR] Download failed from all sources")
        sys.exit(1)

    # Verify the dataset has expected files
    csv_path = data_directory / "ptbxl_database.csv"
    if not csv_path.exists():
        print(f"[ERROR] Expected file not found: {csv_path}")
        print(f"[INFO] Contents of {data_directory}:")
        for item in sorted(data_directory.iterdir())[:20]:
            print(f"  {item.name}")
        sys.exit(1)

    download_marker.touch()
    print(f"[OK] PTB-XL dataset ready at: {data_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PTB-XL dataset")
    parser.add_argument(
        "--source",
        choices=["auto", "kaggle", "physionet"],
        default="auto",
        help="Download source (default: auto — tries Kaggle first, then PhysioNet)",
    )
    args = parser.parse_args()
    download_ptbxl_dataset(source=args.source)
