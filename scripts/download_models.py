# Download ECGFounder model checkpoint from HuggingFace
# Saves the 12-lead ECGFounder weights (~370 MB) to models/ecgfounder/base/

import sys
from pathlib import Path

# Resolve project root relative to this script's location
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def download_ecgfounder_checkpoint() -> None:
    """Download the 12-lead ECGFounder checkpoint from HuggingFace.

    Uses huggingface_hub to fetch the file. Skips download if the
    checkpoint already exists on disk.
    """
    model_directory: Path = PROJECT_ROOT / "models" / "ecgfounder" / "base"
    checkpoint_filename: str = "12_lead_ECGFounder.pth"
    checkpoint_path: Path = model_directory / checkpoint_filename

    # Skip if already downloaded
    if checkpoint_path.exists():
        print(f"[OK] Checkpoint already exists: {checkpoint_path}")
        return

    # Ensure the target directory exists
    model_directory.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading ECGFounder 12-lead checkpoint (~370 MB)...")
    print(f"[INFO] Source: HuggingFace repo PKUDigitalHealth/ECGFounder")
    print(f"[INFO] Destination: {checkpoint_path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed.")
        print("        Install it with: pip install 'huggingface_hub>=0.20.0'")
        sys.exit(1)

    try:
        downloaded_path: str = hf_hub_download(
            repo_id="PKUDigitalHealth/ECGFounder",
            filename=checkpoint_filename,
            local_dir=str(model_directory),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Download complete: {downloaded_path}")
    except Exception as error:
        print(f"[ERROR] Download failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    download_ecgfounder_checkpoint()
