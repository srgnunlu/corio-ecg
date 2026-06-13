# Phase 2 setup script — clones external repos and installs dependencies
# Run from project root: python scripts/setup_phase2.py

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
OPEN_ECG_DIGITIZER_PATCH = PROJECT_ROOT / "vendor" / "patches" / "open-ecg-digitizer.patch"
OPEN_ECG_DIGITIZER_REVISION = "963387ff5abdfa3db91c15ac52d6cf1214345a6a"
ECG_IMAGE_KIT_REVISION = "27b90f56896c9fc78b05a83ca14844ea2637aa0b"


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    """Run a shell command and return its exit code."""
    print(f"  > {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check).returncode


def apply_open_ecg_digitizer_patch(repo_dir: Path, patch_path: Path) -> None:
    """Apply the Corio vendor patch once, failing on incompatible upstream code."""
    if not patch_path.exists():
        raise FileNotFoundError(f"Vendor patch not found: {patch_path}")

    reverse_check = run(
        ["git", "apply", "--reverse", "--check", str(patch_path)],
        cwd=repo_dir,
        check=False,
    )
    if reverse_check == 0:
        print("[OK] Corio Open-ECG-Digitizer patch already applied")
        return

    forward_check = run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=repo_dir,
        check=False,
    )
    if forward_check != 0:
        raise RuntimeError(
            "Open-ECG-Digitizer vendor patch is incompatible with the checked-out revision"
        )

    run(["git", "apply", str(patch_path)], cwd=repo_dir)
    print("[PATCH] Applied Corio Open-ECG-Digitizer compatibility patch")


def ensure_git_revision(repo_dir: Path, revision: str) -> None:
    """Pin an external repository without overwriting local modifications."""
    current_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
    ).strip()
    if current_revision == revision:
        return

    dirty_files = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        text=True,
    ).strip()
    if dirty_files:
        raise RuntimeError(
            f"Cannot pin {repo_dir.name} to {revision}: repository has local changes"
        )
    run(["git", "checkout", revision], cwd=repo_dir)


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

    ensure_git_revision(repo_dir, OPEN_ECG_DIGITIZER_REVISION)
    apply_open_ecg_digitizer_patch(repo_dir, OPEN_ECG_DIGITIZER_PATCH)

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
            [
                "git",
                "clone",
                "https://github.com/alphanumericslab/ecg-image-kit.git",
                str(repo_dir),
            ],
        )
    ensure_git_revision(repo_dir, ECG_IMAGE_KIT_REVISION)


def install_dependencies() -> None:
    """Install additional Python dependencies for Phase 2."""
    packages = ["yacs", "torch-tps"]
    print("[PIP] Installing Phase 2 dependencies...")
    run([sys.executable, "-m", "pip", "install"] + packages)


def verify_installation() -> None:
    """Quick smoke test to verify everything works."""
    print("\n[VERIFY] Running smoke tests...")

    # Test PyTorch + best available device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
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
