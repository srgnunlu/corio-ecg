# Paper ECG image to digital signal conversion via Open-ECG-Digitizer
# Wraps the InferenceWrapper from Ahus-AIM/Open-ECG-Digitizer to produce
# a z-score normalized (12, 5000) numpy array ready for ECGFounder inference

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_torch
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Open-ECG-Digitizer outputs signal in microvolts — ECGFounder expects mV
UV_TO_MV: float = 1000.0

TARGET_SAMPLE_RATE: int = 500
TARGET_LENGTH: int = 5000  # 10 seconds at 500 Hz
NUM_LEADS: int = 12

LEAD_NAMES: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Lead Name U-Net and segmentation models need sufficient resolution
# to accurately detect lead label text and separate signal traces.
# Images smaller than this on the shortest side are upscaled with
# bilinear interpolation before processing.
MIN_IMAGE_DIMENSION: int = 1800


@dataclass
class DigitizeInfo:
    """Diagnostic information from a single digitization run.

    Helps debug lead assignment, amplitude scaling, and layout matching issues.
    """

    image_size: tuple[int, int] = (0, 0)  # (H, W) pixels
    raw_lines_count: int = 0
    detected_leads: list[str] = field(default_factory=list)
    detected_leads_count: int = 0
    layout_name: str = "unknown"
    layout_cost: float = -1.0
    layout_flipped: bool = False
    pixel_spacing_x_mm: float = 0.0
    pixel_spacing_y_mm: float = 0.0
    avg_pixel_per_mm: float = 0.0
    canonical_shape: tuple[int, ...] = (0, 0)
    per_lead_energy: list[float] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        """True if any diagnostic metric suggests a problem."""
        return (
            self.layout_cost > 1.0
            or self.detected_leads_count < 4
            or self.avg_pixel_per_mm < 1.0
            or self.avg_pixel_per_mm > 30.0
            or self.raw_lines_count < 3
        )

    def summary(self) -> str:
        """Human-readable summary for terminal logging."""
        lines = [
            f"Image: {self.image_size[1]}x{self.image_size[0]} px",
            f"Layout: {self.layout_name} (cost: {self.layout_cost:.2f}, "
            f"flipped: {'yes' if self.layout_flipped else 'no'})",
            f"Detected leads: {self.detected_leads_count}/12 "
            f"({', '.join(self.detected_leads) if self.detected_leads else 'none'})",
            f"Raw signal lines: {self.raw_lines_count}",
            f"Pixel spacing: x={self.pixel_spacing_x_mm:.3f} mm/px, "
            f"y={self.pixel_spacing_y_mm:.3f} mm/px "
            f"(avg={self.avg_pixel_per_mm:.1f} px/mm)",
            f"Canonical shape: {self.canonical_shape}",
        ]
        return "\n".join(lines)

def _select_digitiser_device(requested: torch.device | None) -> torch.device:
    """Pick the best device for the digitiser.

    CUDA works correctly at full image resolution. MPS has a resize bug that
    destroys lead label text detail, so it falls back to CPU.
    """
    if requested and requested.type == "cuda":
        if torch.cuda.is_available():
            logger.info("Digitiser using CUDA (full GPU acceleration)")
            return requested
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")

    if requested and requested.type == "mps":
        logger.info("MPS requested but breaks lead detection — forcing CPU")
        return torch.device("cpu")

    # Auto-detect: prefer CUDA > CPU (skip MPS)
    if requested is None:
        if torch.cuda.is_available():
            logger.info("Auto-detected CUDA — digitiser will use GPU")
            return torch.device("cuda")
        logger.info("No CUDA available — digitiser using CPU")
        return torch.device("cpu")

    return requested


# Repo root of the cloned Open-ECG-Digitizer, needed for its internal imports
_DIGITIZER_REPO_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "external" / "open-ecg-digitizer"
)
_DIGITIZER_CONFIG_PATH: Path = (
    _DIGITIZER_REPO_ROOT / "src" / "config" / "inference_wrapper_george-moody-2024.yml"
)


class ECGDigitiser:
    """Converts paper ECG photographs to digital signals using Open-ECG-Digitizer.

    The pipeline: image -> U-Net segmentation -> signal extraction -> lead
    identification -> canonical 12-lead signal in mV -> resample to 500 Hz
    -> z-score normalize -> (12, 5000) numpy array.

    Device policy:
    - CUDA: full GPU acceleration (works correctly at full resolution)
    - MPS: forced to CPU (MPS image resize destroys lead label text detail)
    - CPU: always works, slowest option
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
            device: Torch device override. MPS is forced to CPU because its
                image resize operation degrades lead detection quality.
                CUDA works correctly at full resolution.
        """
        self.device = _select_digitiser_device(device)
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

            # Ensure full layout library is used. The george-moody-2024 config
            # already has good 3×4 layouts, but fall back to all if needed.
            layout_cfg = cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER
            current_path = getattr(layout_cfg, "config_path", "")
            if "reduced" in str(current_path):
                layout_cfg.config_path = "src/config/lead_layouts_all.yml"

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

        self.last_info = DigitizeInfo()

        image_tensor = self._load_image(image_path)
        self.last_info.image_size = (image_tensor.shape[2], image_tensor.shape[3])

        raw_result = self._run_inference(image_tensor)
        self._populate_diagnostics(raw_result)

        canonical = self._extract_canonical(raw_result)
        self.last_info.canonical_shape = tuple(canonical.shape)

        signal = self._postprocess(canonical)

        # Per-lead energy (RMS) after z-score normalization
        for i in range(min(signal.shape[0], NUM_LEADS)):
            rms = float(np.sqrt(np.mean(signal[i] ** 2)))
            self.last_info.per_lead_energy.append(round(rms, 3))

        logger.info("Digitization diagnostics:\n%s", self.last_info.summary())
        if self.last_info.has_warnings:
            logger.warning("Digitization quality concerns detected — see diagnostics")

        return signal

    def _populate_diagnostics(self, raw_result: dict) -> None:
        """Extract diagnostic info from the raw inference result."""
        info = self.last_info

        info.layout_name = raw_result.get("layout_name", "unknown")

        signal_dict = raw_result.get("signal", {})
        info.layout_cost = signal_dict.get("layout_matching_cost", -1.0)
        info.layout_flipped = str(signal_dict.get("layout_is_flipped", "False")) == "True"

        raw_lines = signal_dict.get("raw_lines")
        if raw_lines is not None:
            info.raw_lines_count = raw_lines.shape[0]

        pixel_info = raw_result.get("pixel_spacing_mm", {})
        info.pixel_spacing_x_mm = pixel_info.get("x", 0.0)
        info.pixel_spacing_y_mm = pixel_info.get("y", 0.0)
        info.avg_pixel_per_mm = pixel_info.get("average_pixel_per_mm", 0.0)

        # Detected lead names from the Lead Name U-Net
        detected_names = signal_dict.get("detected_lead_names", [])
        info.detected_leads = list(detected_names)
        info.detected_leads_count = signal_dict.get("n_detected", len(detected_names))

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load image as (1, 3, H, W) tensor, upscaling if too small.

        Lead Name U-Net needs ~1800+ px on the shortest side to reliably
        detect lead label text. Smaller images get bilinear upscaling
        followed by mild sharpening to restore edge detail.
        """
        from torchvision.io import decode_image

        image = decode_image(str(image_path), mode="RGB")
        h, w = image.shape[1], image.shape[2]

        if min(h, w) < MIN_IMAGE_DIMENSION:
            scale = MIN_IMAGE_DIMENSION / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            original_dtype = image.dtype
            image = F_torch.interpolate(
                image.unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(original_dtype)
            # Counteract bilinear upscaling blur with mild sharpening
            image = _sharpen_image(image, strength=0.3)
            logger.info(
                "Upscaled small image: %dx%d -> %dx%d (%.1fx) + sharpened",
                w, h, new_w, new_h, scale,
            )

        logger.debug("Image for processing: %dx%d", image.shape[2], image.shape[1])
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


def _sharpen_image(image: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Apply mild unsharp mask to counteract bilinear upscaling blur.

    Only used on upscaled images — sharpens ECG trace edges and lead
    label text without amplifying grid line noise excessively.

    Args:
        image: (3, H, W) uint8 tensor.
        strength: Blend factor (0 = no effect, 1 = full sharpening).
    """
    # 3x3 Gaussian blur kernel applied per channel
    kernel = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32,
    ) / 16.0
    kernel = kernel.reshape(1, 1, 3, 3).expand(3, 1, 3, 3)

    img_float = image.float().unsqueeze(0)  # (1, 3, H, W)
    blurred = F_torch.conv2d(img_float, kernel, padding=1, groups=3)
    sharpened = img_float + strength * (img_float - blurred)
    return sharpened.clamp(0, 255).squeeze(0).to(image.dtype)


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
