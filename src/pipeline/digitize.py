# Paper ECG image to digital signal conversion via Open-ECG-Digitizer
# Wraps the InferenceWrapper from Ahus-AIM/Open-ECG-Digitizer to produce
# a z-score normalized (12, 5000) numpy array ready for ECGFounder inference

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

from src.pipeline.diagnose import get_device

logger = logging.getLogger(__name__)

# Open-ECG-Digitizer outputs signal in microvolts — ECGFounder expects mV
UV_TO_MV: float = 1000.0

TARGET_SAMPLE_RATE: int = 500
TARGET_LENGTH: int = 5000  # 10 seconds at 500 Hz
NUM_LEADS: int = 12

# Repo root of the cloned Open-ECG-Digitizer, needed for its internal imports
_DIGITIZER_REPO_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "external" / "open-ecg-digitizer"
)
_DIGITIZER_CONFIG_PATH: Path = (
    _DIGITIZER_REPO_ROOT / "src" / "config" / "inference_wrapper.yml"
)


class ECGDigitiser:
    """Converts paper ECG photographs to digital signals using Open-ECG-Digitizer.

    The pipeline: image -> U-Net segmentation -> signal extraction -> lead
    identification -> canonical 12-lead signal in mV -> resample to 500 Hz
    -> z-score normalize -> (12, 5000) numpy array.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the digitiser by loading Open-ECG-Digitizer's InferenceWrapper.

        Args:
            config_path: Path to inference_wrapper.yml. Defaults to the one
                shipped with the cloned repo.
            device: Torch device override. Auto-detected if None.
        """
        self.device = device or get_device()
        self.config_path = Path(config_path) if config_path else _DIGITIZER_CONFIG_PATH
        self._wrapper = self._load_wrapper()
        logger.info("ECGDigitiser ready — device=%s", self.device)

    def _load_wrapper(self) -> torch.nn.Module:
        """Load InferenceWrapper with device overrides applied.

        The Open-ECG-Digitizer uses `src.*` imports internally, which clash
        with our own `src` package. We temporarily hijack sys.path and
        invalidate import caches so Python resolves their `src` not ours.
        """
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        # Stash any cached 'src' modules from our project so the digitizer's
        # `src.*` imports resolve to its own package tree
        stashed_modules = {
            key: sys.modules.pop(key)
            for key in list(sys.modules)
            if key == "src" or key.startswith("src.")
        }

        try:
            repo_str = str(_DIGITIZER_REPO_ROOT)
            sys.path = [repo_str] + [p for p in sys.path if p != repo_str]

            from src.config.default import get_cfg  # type: ignore[import-untyped]
            from src.model.inference_wrapper import InferenceWrapper  # type: ignore[import-untyped]

            # Open-ECG-Digitizer uses relative paths in its config (e.g.
            # ./weights/...), so we must run from its repo root
            os.chdir(_DIGITIZER_REPO_ROOT)

            cfg = get_cfg(str(self.config_path))
            # Override device everywhere — the YAML defaults to 'cuda'
            device_str = str(self.device)
            cfg.MODEL.KWARGS.device = device_str
            cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device = device_str

            wrapper: InferenceWrapper = InferenceWrapper(**cfg.MODEL.KWARGS)
            # Skip wrapper.eval() — the InferenceWrapper stores non-Module
            # objects (Dewarper, Cropper) as attributes, which breaks PyTorch's
            # recursive eval(). The segmentation UNet is already set to eval
            # mode inside _load_segmentation_model().
        finally:
            os.chdir(original_cwd)
            sys.path = original_path
            # Remove the digitizer's src modules and restore ours
            for key in list(sys.modules):
                if key == "src" or key.startswith("src."):
                    sys.modules.pop(key, None)
            sys.modules.update(stashed_modules)

        return wrapper

    def digitize(self, image_path: str | Path) -> np.ndarray:
        """Convert a paper ECG image to a 12-lead digital signal.

        Args:
            image_path: Path to ECG image (PNG/JPG).

        Returns:
            Z-score normalized numpy array of shape (12, 5000) at 500 Hz,
            ready for ECGFounder inference.

        Raises:
            FileNotFoundError: If image_path does not exist.
            RuntimeError: If digitization fails (e.g. no signal detected).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_tensor = self._load_image(image_path)
        raw_result = self._run_inference(image_tensor)
        canonical = self._extract_canonical(raw_result)
        signal = self._postprocess(canonical)
        return signal

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load image as (1, 3, H, W) float tensor."""
        from torchvision.io import decode_image

        image = decode_image(str(image_path), mode="RGB")
        return image.unsqueeze(0)

    def _run_inference(self, image_tensor: torch.Tensor) -> dict:
        """Run Open-ECG-Digitizer forward pass."""
        # layout_should_include_substring=None means auto-detect layout
        result: dict = self._wrapper(image_tensor, layout_should_include_substring=None)
        return result

    def _extract_canonical(self, result: dict) -> torch.Tensor:
        """Extract canonical_lines tensor from inference result.

        Raises:
            RuntimeError: If canonical_lines is missing or None (detection failed).
        """
        signal_dict = result.get("signal", {})
        canonical = signal_dict.get("canonical_lines")
        if canonical is None:
            raise RuntimeError(
                "Open-ECG-Digitizer could not extract canonical leads. "
                "The image may be too noisy or not a standard ECG layout."
            )
        return canonical

    @staticmethod
    def _postprocess(canonical: torch.Tensor) -> np.ndarray:
        """Convert raw canonical_lines to ECGFounder-ready format.

        Steps: NaN->0, uV->mV, resample to 500Hz/5000pts, pad/truncate, z-score.
        """
        signal = canonical.cpu().numpy().astype(np.float64)

        # Handle NaN values (overlapping/undetected leads produce NaN)
        signal = np.nan_to_num(signal, nan=0.0)

        # Convert microvolts to millivolts
        signal = signal / UV_TO_MV

        # Ensure exactly 12 leads
        signal = _pad_or_truncate_leads(signal, NUM_LEADS)

        # Resample to 500 Hz / 5000 samples
        n_points = signal.shape[1]
        if n_points != TARGET_LENGTH:
            signal = _resample_signal(signal, n_points, TARGET_LENGTH)

        # Pad or truncate time axis to exactly 5000 samples
        signal = _pad_or_truncate_time(signal, TARGET_LENGTH)

        # Global z-score normalization (same as wfdb_helpers)
        signal = _z_score_normalize(signal)

        return signal.astype(np.float32)


def _pad_or_truncate_leads(signal: np.ndarray, target_leads: int) -> np.ndarray:
    """Ensure signal has exactly target_leads rows."""
    n_leads = signal.shape[0]
    if n_leads >= target_leads:
        return signal[:target_leads]
    padded = np.zeros((target_leads, signal.shape[1]), dtype=signal.dtype)
    padded[:n_leads] = signal
    return padded


def _resample_signal(
    signal: np.ndarray, source_length: int, target_length: int
) -> np.ndarray:
    """Resample each lead from source_length to target_length points.

    Uses endpoint=True so source and target share the same [0, 1] range,
    avoiding out-of-bounds interpolation when upsampling.
    """
    source_times = np.linspace(0, 1, source_length, endpoint=True)
    target_times = np.linspace(0, 1, target_length, endpoint=True)
    resampled = np.zeros((signal.shape[0], target_length), dtype=signal.dtype)
    for lead_idx in range(signal.shape[0]):
        interpolator = interp1d(source_times, signal[lead_idx], kind="linear")
        resampled[lead_idx] = interpolator(target_times)
    return resampled


def _pad_or_truncate_time(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or truncate to target_length samples on time axis."""
    n_points = signal.shape[1]
    if n_points >= target_length:
        return signal[:, :target_length]
    padded = np.zeros((signal.shape[0], target_length), dtype=signal.dtype)
    padded[:, :n_points] = signal
    return padded


def _z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """Global z-score normalization (zero mean, unit variance)."""
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)
