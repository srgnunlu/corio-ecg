# Synthetic paper ECG image generator — wraps ECG-Image-Kit as a subprocess
# to render WFDB records as realistic paper ECG photographs with configurable
# difficulty levels (clean, moderate, hard) for training the digitizer.

import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

# Absolute path to the ECG-Image-Kit generator directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ECG_IMAGE_KIT_DIR = _PROJECT_ROOT / "external" / "ecg-image-kit" / "codes" / "ecg-image-generator"
_GENERATOR_SCRIPT = _ECG_IMAGE_KIT_DIR / "gen_ecg_image_from_data.py"

# imgaug is abandoned and incompatible with NumPy 2.x (np.sctypes removed).
# This inline launcher patches numpy before importing the generator script,
# so we don't need to modify the vendored ECG-Image-Kit code.
_NUMPY_COMPAT_LAUNCHER = (
    "import numpy as np; "
    "np.sctypes = {"
    "'int': [np.int8, np.int16, np.int32, np.int64], "
    "'uint': [np.uint8, np.uint16, np.uint32, np.uint64], "
    "'float': [np.float16, np.float32, np.float64], "
    "'complex': [np.complex64, np.complex128], "
    "'others': [bool, object, bytes, str, np.void]"
    "}; "
    "np.bool = np.bool_; "
    "np.int = np.int_; "
    "np.float = np.float64; "
    "np.complex = np.complex128; "
    "np.object = np.object_; "
    "np.str = np.str_; "
    # Mock unused ECG-Image-Kit subpackages that have heavy/unavailable deps.
    # HandwrittenText needs tensorflow+seaborn, CreasesWrinkles needs imutils
    # and crashes on OpenCV 4.13+. We don't use --hw_text or --wrinkles.
    "import types, sys; "
    "sys.modules['HandwrittenText'] = types.ModuleType('HandwrittenText'); "
    "ht_gen = types.ModuleType('HandwrittenText.generate'); "
    "ht_gen.get_handwritten = lambda *a, **k: None; "
    "sys.modules['HandwrittenText.generate'] = ht_gen; "
    "sys.modules['CreasesWrinkles'] = types.ModuleType('CreasesWrinkles'); "
    "cw = types.ModuleType('CreasesWrinkles.creases'); "
    "cw.get_creased = lambda *a, **k: None; "
    "sys.modules['CreasesWrinkles.creases'] = cw; "
    f"import runpy; runpy.run_path('{_GENERATOR_SCRIPT}', run_name='__main__')"
)


class DifficultyLevel(Enum):
    """Controls how much visual noise is added to the synthetic ECG image.

    CLEAN: Grid lines only, no distortions — ideal for initial testing.
    MODERATE: Light augmentation (rotation, noise, color shift).
    HARD: Heavy distortions (wrinkles, creases, handwritten text, strong noise).
    """

    CLEAN = "clean"
    MODERATE = "moderate"
    HARD = "hard"


def _build_cli_args(
    dat_path: str,
    hea_path: str,
    output_dir: str,
    difficulty: DifficultyLevel,
    seed: int,
) -> list[str]:
    """Build the CLI argument list for gen_ecg_image_from_data.py.

    Each difficulty level maps to a fixed set of flags so the rendering
    is reproducible and consistent across the dataset.
    """
    base_args = [
        "python", "-c", _NUMPY_COMPAT_LAUNCHER,
        "-i", dat_path,
        "-hea", hea_path,
        "-o", output_dir,
        "-se", str(seed),
        "-st", "0",
        "--num_leads", "twelve",
    ]

    if difficulty == DifficultyLevel.CLEAN:
        base_args += [
            "-r", "200",
            "--standard_grid_color", "5",
        ]

    elif difficulty == DifficultyLevel.MODERATE:
        # --store_config 2 is required when --augment is used, because
        # get_augment() reads lead bbox info from the JSON config
        base_args += [
            "-r", "200",
            "--standard_grid_color", "5",
            "--store_config", "2",
            "--augment",
            "-noise", "25",
            "-rot", "2",
            "-c", "0.01",
            "-t", "8000",
        ]

    elif difficulty == DifficultyLevel.HARD:
        # --wrinkles removed: ECG-Image-Kit's CreasesWrinkles module uses
        # cv2.subtract with incompatible types on OpenCV 4.13+.
        # --hw_text removed: requires spacy en_core_sci_sm model (~200 MB).
        # Instead, use aggressive augmentation (heavy noise, rotation, crop,
        # color temperature shift) to simulate difficult scanning conditions.
        base_args += [
            "-r", "150",
            "--standard_grid_color", "5",
            "--store_config", "2",
            "--random_grid_color",
            "--augment",
            "-noise", "50",
            "-rot", "8",
            "-c", "0.03",
            "-t", "3000",
        ]

    return base_args


def _find_generated_png(output_dir: Path, record_stem: str) -> Optional[Path]:
    """Locate the PNG file produced by ECG-Image-Kit.

    The generator names output files as '{record_stem}-0.png'.
    We also handle cases where the suffix might differ.
    """
    # Primary expected name
    expected = output_dir / f"{record_stem}-0.png"
    if expected.exists():
        return expected

    # Fallback: find any PNG that starts with the record stem
    matches = sorted(output_dir.glob(f"{record_stem}*.png"))
    return matches[0] if matches else None


def render_ecg_image(
    record_path: str,
    output_path: str | Path,
    difficulty: DifficultyLevel = DifficultyLevel.CLEAN,
    seed: int = 42,
) -> Path:
    """Render a WFDB ECG record as a synthetic paper ECG image.

    Args:
        record_path: WFDB record path without extension (e.g.
            'data/raw/ptb-xl/records500/00000/00001_hr').
        output_path: Destination path for the final PNG file.
        difficulty: Visual noise level (CLEAN, MODERATE, HARD).
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated PNG image.

    Raises:
        FileNotFoundError: If .dat or .hea files are missing.
        RuntimeError: If ECG-Image-Kit script fails or produces no output.
    """
    output_path = Path(output_path)
    record_path_obj = Path(record_path)

    dat_file = record_path_obj.with_suffix(".dat")
    hea_file = record_path_obj.with_suffix(".hea")

    if not dat_file.exists():
        raise FileNotFoundError(f"DAT file not found: {dat_file}")
    if not hea_file.exists():
        raise FileNotFoundError(f"HEA file not found: {hea_file}")

    # Use a temp directory so we don't pollute the source tree
    with tempfile.TemporaryDirectory(prefix="ecg_render_") as tmp_dir:
        cli_args = _build_cli_args(
            dat_path=str(dat_file.resolve()),
            hea_path=str(hea_file.resolve()),
            output_dir=tmp_dir,
            difficulty=difficulty,
            seed=seed,
        )

        # ECG-Image-Kit uses os.getcwd() and relative imports, so we
        # must run the subprocess from its own directory
        result = subprocess.run(
            cli_args,
            cwd=str(_ECG_IMAGE_KIT_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ECG-Image-Kit failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )

        # Locate the generated PNG in the temp directory
        record_stem = record_path_obj.stem
        generated_png = _find_generated_png(Path(tmp_dir), record_stem)

        if generated_png is None:
            raise RuntimeError(
                f"ECG-Image-Kit produced no PNG for '{record_stem}'. "
                f"Files in output dir: {list(Path(tmp_dir).iterdir())}"
            )

        # Move the result to the requested output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated_png), str(output_path))

    return output_path
