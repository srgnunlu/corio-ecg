# Download PTB-XL dataset from PhysioNet
# Fetches the zip archive (~3 GB), extracts it, and organizes files into data/raw/ptb-xl/

import shutil
import subprocess
import sys
from pathlib import Path

# Resolve project root relative to this script's location
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

PTBXL_URL: str = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
PTBXL_ZIP_FILENAME: str = "ptb-xl-1.0.3.zip"

# The zip extracts into a nested directory with this name
PTBXL_NESTED_DIRECTORY_NAME: str = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def run_command(command: list[str], description: str) -> None:
    """Run a shell command and exit on failure.

    Args:
        command: Command and arguments to execute.
        description: Human-readable description shown on failure.
    """
    print(f"[INFO] Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed:")
        print(result.stderr)
        sys.exit(1)


def download_ptbxl_dataset() -> None:
    """Download and extract the PTB-XL dataset from PhysioNet.

    Uses curl for downloading and unzip for extraction. A .download_complete
    marker file is used to skip re-download on subsequent runs.
    """
    data_directory: Path = PROJECT_ROOT / "data" / "raw" / "ptb-xl"
    download_marker: Path = data_directory / ".download_complete"
    zip_file_path: Path = data_directory / PTBXL_ZIP_FILENAME

    # Skip if already downloaded and extracted
    if download_marker.exists():
        print(f"[OK] PTB-XL dataset already downloaded: {data_directory}")
        return

    # Ensure the target directory exists
    data_directory.mkdir(parents=True, exist_ok=True)

    # Download the zip file using curl
    print(f"[INFO] Downloading PTB-XL dataset (~3 GB)...")
    print(f"[INFO] Source: {PTBXL_URL}")
    print(f"[INFO] Destination: {zip_file_path}")

    run_command(
        ["curl", "-L", "-o", str(zip_file_path), "--progress-bar", PTBXL_URL],
        description="Download",
    )
    print("[OK] Download complete.")

    # Extract the zip file
    print(f"[INFO] Extracting archive...")
    run_command(
        ["unzip", "-q", "-o", str(zip_file_path), "-d", str(data_directory)],
        description="Extraction",
    )
    print("[OK] Extraction complete.")

    # Move files from the nested directory to data_directory
    nested_directory: Path = data_directory / PTBXL_NESTED_DIRECTORY_NAME
    if nested_directory.exists():
        print(f"[INFO] Moving files from nested directory to {data_directory}...")
        for item in nested_directory.iterdir():
            destination: Path = data_directory / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
        # Remove the now-empty nested directory
        nested_directory.rmdir()
        print("[OK] Files reorganized.")

    # Clean up the zip file to save disk space
    if zip_file_path.exists():
        print(f"[INFO] Removing zip file to save space...")
        zip_file_path.unlink()
        print("[OK] Zip file removed.")

    # Create marker file to indicate successful download
    download_marker.touch()
    print(f"[OK] PTB-XL dataset ready at: {data_directory}")


if __name__ == "__main__":
    download_ptbxl_dataset()
