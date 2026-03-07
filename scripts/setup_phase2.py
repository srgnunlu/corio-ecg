# Phase 2 setup script — clones external repos and installs dependencies
# Run from project root: python scripts/setup_phase2.py

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a shell command, raising on failure."""
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def setup_open_ecg_digitizer() -> None:
    """Clone Open-ECG-Digitizer and pull model weights."""
    repo_dir = EXTERNAL_DIR / "open-ecg-digitizer"

    if repo_dir.exists():
        print("[OK] Open-ECG-Digitizer already cloned")
    else:
        print("[CLONE] Open-ECG-Digitizer...")
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "https://github.com/Ahus-AIM/Open-ECG-Digitizer.git", str(repo_dir)],
        )

    # Verify model weights were downloaded via LFS
    weights_dir = repo_dir / "weights"
    unet_weights = weights_dir / "unet_weights_07072025.pt"
    if unet_weights.exists() and unet_weights.stat().st_size > 1_000_000:
        print("[OK] Model weights present")
    else:
        print("[LFS] Pulling model weights...")
        run(["git", "lfs", "pull"], cwd=repo_dir)


def setup_ecg_image_kit() -> None:
    """Clone ECG-Image-Kit for synthetic image generation."""
    repo_dir = EXTERNAL_DIR / "ecg-image-kit"

    if repo_dir.exists():
        print("[OK] ECG-Image-Kit already cloned")
    else:
        print("[CLONE] ECG-Image-Kit...")
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "https://github.com/alphanumericslab/ecg-image-kit.git", str(repo_dir)],
        )


def install_dependencies() -> None:
    """Install additional Python dependencies for Phase 2."""
    packages = ["yacs", "torch-tps"]
    print("[PIP] Installing Phase 2 dependencies...")
    run([sys.executable, "-m", "pip", "install"] + packages)


def verify_installation() -> None:
    """Quick smoke test to verify everything works."""
    print("\n[VERIFY] Running smoke tests...")

    # Test PyTorch + MPS
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  PyTorch {torch.__version__} — device: {device}")

    # Test Open-ECG-Digitizer UNet loads
    sys.path.insert(0, str(EXTERNAL_DIR / "open-ecg-digitizer"))
    from src.model.unet import UNet

    model = UNet(
        num_in_channels=3, num_out_channels=4,
        dims=[32, 64, 128, 256, 320, 320, 320, 320], depth=2,
    )
    weights_path = EXTERNAL_DIR / "open-ecg-digitizer" / "weights" / "unet_weights_07072025.pt"
    checkpoint = torch.load(weights_path, weights_only=True, map_location=device)
    checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    print("  Open-ECG-Digitizer UNet loaded OK")

    print("\n[DONE] Phase 2 setup complete!")


def main() -> None:
    print("=" * 50)
    print("Corio ECG — Phase 2 Setup")
    print("=" * 50)

    setup_open_ecg_digitizer()
    setup_ecg_image_kit()
    install_dependencies()
    verify_installation()


if __name__ == "__main__":
    main()
