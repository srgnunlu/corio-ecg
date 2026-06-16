# Paper ECG image to digital signal conversion via Open-ECG-Digitizer
# Wraps the InferenceWrapper from Ahus-AIM/Open-ECG-Digitizer to produce
# a z-score normalized (12, 5000) numpy array ready for ECGFounder inference

from __future__ import annotations

import gc
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F_torch
from scipy.interpolate import interp1d

from src.pipeline.perspective_correction import (
    PERSPECTIVE_CORRECTION_VERSION,
    correct_perspective,
)
from src.pipeline.shadow_normalization import (
    SHADOW_NORMALIZATION_VERSION,
    normalize_broad_shadow,
)
from src.utils.signal_clean import (
    einthoven_consistency,
    highpass_filter,
    wavelet_denoise,
)

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
# The digitizer was trained at ~1800px shortest side. Without upscaling,
# grid detection and signal separation fail (tested: 2 lines at 1700px
# vs 4 lines at 1800px). MPS segmentation handles the memory cost.
MIN_IMAGE_DIMENSION: int = 1800
MAX_IMAGE_DIMENSION: int = 2400
MIN_REQUIRED_RAW_LINES: int = 3
MIN_REQUIRED_NONZERO_LEADS: int = 8
LEAD_ACTIVITY_THRESHOLD: float = 5e-2


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
    nonzero_leads_count: int = 0
    einthoven_score: float = -1.0  # Pearson corr: II vs I+III (-1=not computed)
    processing_mode: str = "default"
    page_correction_applied: bool = False
    page_correction_confidence: float = 0.0
    page_correction_version: str = "disabled"
    shadow_normalization_applied: bool = False
    shadow_score: float = 0.0
    shadow_normalization_version: str = "disabled"
    # Re-projection fidelity artifacts (image-space ink overlay). Populated only
    # when the digitizer exposes them; large arrays kept off the summary output.
    signal_probability: np.ndarray | None = None  # (H, W) segmentation ink map
    raw_lines: np.ndarray | None = None  # (n_traces, W) extracted pixel-Y traces
    extraction_crop_x0: int = 0  # leading-column crop mapping raw_lines -> ink frame
    # Full-duration rhythm strip for HR analysis (uncropped, z-scored). None
    # when the layout prints no full-width rhythm lead.
    rhythm_strip: np.ndarray | None = None

    @property
    def has_warnings(self) -> bool:
        """True if any diagnostic metric suggests a problem."""
        return (
            self.layout_cost > 1.0
            or self.detected_leads_count < 4
            or self.avg_pixel_per_mm < 1.0
            or self.avg_pixel_per_mm > 30.0
            or self.raw_lines_count < MIN_REQUIRED_RAW_LINES
            or (
                self.nonzero_leads_count > 0
                and self.nonzero_leads_count < MIN_REQUIRED_NONZERO_LEADS
            )
            or (self.einthoven_score != -1.0 and self.einthoven_score < 0.7)
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
            f"Non-zero leads: {self.nonzero_leads_count}/12",
            f"Processing mode: {self.processing_mode}",
            f"Page correction: {'applied' if self.page_correction_applied else 'not applied'} "
            f"(confidence: {self.page_correction_confidence:.2f})",
            f"Pixel spacing: x={self.pixel_spacing_x_mm:.3f} mm/px, "
            f"y={self.pixel_spacing_y_mm:.3f} mm/px "
            f"(avg={self.avg_pixel_per_mm:.1f} px/mm)",
            f"Canonical shape: {self.canonical_shape}",
            f"Einthoven consistency: {self.einthoven_score:.2f}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class DigitizedSignals:
    """Diagnosis-ready and calibrated outputs from one digitization run."""

    model_input: np.ndarray
    calibrated_millivolts: np.ndarray
    # Full-duration rhythm strip (uncropped full-width leads, others zeroed),
    # z-score normalized like model_input. Preserves the true 10 s RR sequence
    # for heart-rate analysis. None when the layout has no full-width strip.
    rhythm_strip: np.ndarray | None = None


def _mock_ray_tune_if_missing() -> None:
    """Inject a fake ray.tune module if ray is not installed.

    Open-ECG-Digitizer's utils.py imports ray.tune.Stopper for its
    EarlyStopper training class. We never use training — only inference —
    so a minimal mock prevents the ImportError without installing ray
    (which pulls in pyarrow and many heavy dependencies).
    """
    if "ray" not in sys.modules:
        try:
            import ray  # noqa: F401
        except ImportError:
            import types

            ray_mod = types.ModuleType("ray")
            tune_mod = types.ModuleType("ray.tune")

            class _FakeStopper:
                pass

            tune_mod.Stopper = _FakeStopper  # type: ignore[attr-defined]
            ray_mod.tune = tune_mod  # type: ignore[attr-defined]
            sys.modules["ray"] = ray_mod
            sys.modules["ray.tune"] = tune_mod
            logger.debug("Injected ray.tune mock (training-only dependency)")


def _select_digitiser_device(requested: torch.device | None) -> torch.device:
    """Pick the best device for the digitiser's pipeline (CPU for safety).

    The perspective detector and layout identifier have MPS tensor
    indexing bugs, so the main pipeline MUST run on CPU. The segmentation
    U-Net is moved to MPS separately (see _load_wrapper) for memory
    efficiency. CUDA works correctly for everything.
    """
    if requested and requested.type == "cuda":
        if torch.cuda.is_available():
            logger.info("Digitiser using CUDA (full GPU acceleration)")
            return requested
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")

    # MPS has bugs in perspective_detector and layout_identifier indexing.
    # Force main pipeline to CPU; segmentation U-Net is moved to MPS
    # separately in _load_wrapper for memory efficiency.
    if requested and requested.type == "mps":
        logger.info("MPS requested — pipeline on CPU, segmentation on MPS")
        return torch.device("cpu")

    # Auto-detect: prefer CUDA > CPU (MPS used only for segmentation)
    if requested is None:
        if torch.cuda.is_available():
            logger.info("Auto-detected CUDA — digitiser will use GPU")
            return torch.device("cuda")
        logger.info("Digitiser using CPU (segmentation offloaded to MPS if available)")
        return torch.device("cpu")

    return requested


def _crop_likely_landscape_page(image: torch.Tensor) -> torch.Tensor:
    """Crop a textured landscape ECG page embedded in a portrait photo.

    Portrait monitor photos can contain an already-upright landscape ECG page.
    Cropping that page before the general portrait rotation avoids turning the
    ECG sideways. The thresholds are intentionally conservative so ordinary
    paper photos that fill the frame are left unchanged.
    """
    import cv2

    if image.ndim != 3 or image.shape[1] <= image.shape[2]:
        return image

    _, height, width = image.shape
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, bright_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    kernel_size = max(5, round(min(height, width) / 35))
    if kernel_size % 2 == 0:
        kernel_size += 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )
    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    edges = cv2.Canny(gray, 50, 150)

    best_bounds: tuple[int, int, int, int] | None = None
    best_score = 0.0
    total_area = height * width
    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        area_ratio = candidate_width * candidate_height / total_area
        aspect_ratio = candidate_width / max(candidate_height, 1)
        width_ratio = candidate_width / width
        if not (
            0.12 <= area_ratio <= 0.85
            and 1.05 <= aspect_ratio <= 2.2
            and width_ratio >= 0.55
        ):
            continue

        edge_density = float(
            np.mean(edges[y : y + candidate_height, x : x + candidate_width] > 0)
        )
        if edge_density < 0.06:
            continue
        score = area_ratio * edge_density
        if score > best_score:
            best_score = score
            best_bounds = (x, y, candidate_width, candidate_height)

    if best_bounds is None:
        return image

    x, y, candidate_width, candidate_height = best_bounds
    margin = max(2, round(min(height, width) * 0.01))
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(width, x + candidate_width + margin)
    bottom = min(height, y + candidate_height + margin)
    logger.info(
        "Cropped landscape ECG page from portrait photo: %dx%d -> %dx%d",
        width,
        height,
        right - left,
        bottom - top,
    )
    return image[:, top:bottom, left:right].contiguous()


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
        enable_dewarping_retry: bool = True,
        enable_orientation_retry: bool = True,
        enable_perspective_correction: bool = False,
        enable_shadow_normalization: bool = False,
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
        self.enable_dewarping_retry = enable_dewarping_retry
        self.enable_orientation_retry = enable_orientation_retry
        self.enable_perspective_correction = enable_perspective_correction
        self.enable_shadow_normalization = enable_shadow_normalization
        self.config_path = Path(config_path) if config_path else _DIGITIZER_CONFIG_PATH
        self._wrapper = self._load_wrapper()
        logger.info(
            "ECGDigitiser ready — device=%s, dewarping_retry=%s, "
            "orientation_retry=%s, perspective_correction=%s, shadow_normalization=%s",
            self.device,
            self.enable_dewarping_retry,
            self.enable_orientation_retry,
            self.enable_perspective_correction,
            self.enable_shadow_normalization,
        )

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

            # Mock ray.tune — Open-ECG-Digitizer imports it for training only
            # (EarlyStopper class), not needed for inference
            _mock_ray_tune_if_missing()

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

            # Disable wrapper's internal resampling — our _load_image already
            # upscales small images to MIN_IMAGE_DIMENSION. The config default
            # (resample_size=3000) would double-upscale, making images ~5000px
            # and causing 4+ minute processing times.
            cfg.MODEL.KWARGS.resample_size = None

            # Enable timing to log per-stage durations
            cfg.MODEL.KWARGS.enable_timing = True

            # Ensure full layout library is used. The george-moody-2024 config
            # already has good 3×4 layouts, but fall back to all if needed.
            layout_cfg = cfg.MODEL.KWARGS.config.LAYOUT_IDENTIFIER
            current_path = getattr(layout_cfg, "config_path", "")
            if "reduced" in str(current_path):
                layout_cfg.config_path = "src/config/lead_layouts_all.yml"

            wrapper: InferenceWrapper = InferenceWrapper(**cfg.MODEL.KWARGS)

            # Move ONLY the segmentation U-Net to MPS (Apple Metal GPU).
            # Metal's allocator returns freed memory to the OS, preventing
            # the runaway RSS growth from PyTorch CPU im2col temporaries.
            # Everything else stays on CPU to avoid MPS indexing bugs in
            # perspective_detector and layout_identifier.
            if self.device.type != "cuda" and torch.backends.mps.is_available():
                wrapper.segmentation_model = wrapper.segmentation_model.to("mps")
                wrapper._seg_device = torch.device("mps")
                logger.info("Segmentation U-Net moved to MPS for memory efficiency")

            # Cap non-segmentation stages on CPU to limit im2col memory.
            # Signal extraction is NOT capped — it uses pixel tracing (not
            # conv2d), so memory is linear. Full resolution is critical for
            # thin rhythm strip traces that vanish when downscaled.
            if self.device.type != "cuda":
                wrapper._max_crop_width = 1200
                wrapper._max_identifier_width = 1000
        finally:
            os.chdir(original_cwd)
            sys.path = original_path
            # Remove the digitizer's src modules and restore ours
            for key in list(sys.modules):
                if key == "src" or key.startswith("src."):
                    sys.modules.pop(key, None)
            sys.modules.update(stashed_modules)

        return wrapper

    def digitize(
        self,
        image_path: str | Path,
        layout_hint: str | None = None,
    ) -> np.ndarray:
        """Convert a paper ECG image to a 12-lead digital signal.

        Args:
            image_path: Path to ECG image (PNG/JPG).
            layout_hint: Optional layout substring filter (e.g. "3x4", "6x2").
                When None, the digitizer auto-detects layout.

        Returns:
            Z-score normalized numpy array of shape (12, 5000) at 500 Hz,
            ready for ECGFounder inference.

        Raises:
            FileNotFoundError: If image_path does not exist.
            RuntimeError: If digitization fails (e.g. no signal detected).
        """
        return self.digitize_with_calibrated(
            image_path,
            layout_hint=layout_hint,
        ).model_input

    def digitize_with_calibrated(
        self,
        image_path: str | Path,
        layout_hint: str | None = None,
    ) -> DigitizedSignals:
        """Digitize an image and preserve the pre-normalization mV signal."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.last_info = DigitizeInfo()

        image_tensor = self._load_image(image_path)
        self.last_info.image_size = (image_tensor.shape[2], image_tensor.shape[3])

        raw_result = self._run_best_inference(image_tensor, layout_hint=layout_hint)
        del image_tensor
        gc.collect()
        self._populate_diagnostics(raw_result)

        canonical = self._extract_canonical(raw_result)
        self.last_info.canonical_shape = tuple(canonical.shape)

        outputs = self._postprocess_outputs(canonical)
        signal = outputs.model_input
        self.last_info.rhythm_strip = outputs.rhythm_strip
        self.last_info.nonzero_leads_count = self._count_nonzero_leads(signal)

        # Per-lead energy (RMS) after z-score normalization
        for i in range(min(signal.shape[0], NUM_LEADS)):
            rms = float(np.sqrt(np.mean(signal[i] ** 2)))
            self.last_info.per_lead_energy.append(round(rms, 3))

        # Verify lead assignment via Einthoven's law (II ≈ I + III)
        self.last_info.einthoven_score = einthoven_consistency(signal)
        if self.last_info.einthoven_score < 0.7:
            logger.warning(
                "Einthoven consistency low (%.2f) — lead assignment may be incorrect",
                self.last_info.einthoven_score,
            )

        logger.info("Digitization diagnostics:\n%s", self.last_info.summary())
        if self.last_info.has_warnings:
            logger.warning("Digitization quality concerns detected — see diagnostics")

        self._validate_digitized_signal(signal, layout_hint=layout_hint)

        return outputs

    def _populate_diagnostics(self, raw_result: dict) -> None:
        """Extract diagnostic info from the raw inference result."""
        info = self.last_info

        info.layout_name = raw_result.get("layout_name", "unknown")
        info.processing_mode = raw_result.get("processing_mode", "default")

        signal_dict = raw_result.get("signal", {})
        info.layout_cost = signal_dict.get("layout_matching_cost", -1.0)
        info.layout_flipped = str(signal_dict.get("layout_is_flipped", "False")) == "True"

        raw_lines = signal_dict.get("raw_lines")
        if raw_lines is not None:
            info.raw_lines_count = raw_lines.shape[0]
            info.raw_lines = _to_numpy(raw_lines)

        # Re-projection fidelity inputs: the segmentation ink map plus the
        # column-crop offset that aligns raw_lines to that map's frame.
        signal_probability = signal_dict.get("signal_probability")
        if signal_probability is not None:
            info.signal_probability = _to_numpy(signal_probability)
        info.extraction_crop_x0 = int(signal_dict.get("extraction_crop_x0", 0))

        pixel_info = raw_result.get("pixel_spacing_mm", {})
        info.pixel_spacing_x_mm = pixel_info.get("x", 0.0)
        info.pixel_spacing_y_mm = pixel_info.get("y", 0.0)
        info.avg_pixel_per_mm = pixel_info.get("average_pixel_per_mm", 0.0)

        # Detected lead names from the Lead Name U-Net
        detected_names = signal_dict.get("detected_lead_names", [])
        info.detected_leads = list(detected_names)
        info.detected_leads_count = signal_dict.get("n_detected", len(detected_names))

    # Maximum upscale factor — beyond 2x, bilinear interpolation creates
    # too much blur for the U-Net to separate signal from grid reliably.
    _MAX_UPSCALE_FACTOR: float = 2.0

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load image as (1, 3, H, W) tensor, clamped to safe dimensions.

        Applies six pre-processing steps in order:
        1. Crop an already-upright landscape ECG page from portrait screen photos
        2. Auto-rotate remaining portrait images (ECGs are always landscape)
        3. Optionally rectify a high-confidence quadrilateral page
        4. Downscale oversized phone photos to MAX_IMAGE_DIMENSION
        5. Optionally normalize a broad illumination shadow
        6. Upscale small images to MIN_IMAGE_DIMENSION (capped at 2x)
        """
        from torchvision.io import decode_image

        image = decode_image(str(image_path), mode="RGB")
        image = _crop_likely_landscape_page(image)
        h, w = image.shape[1], image.shape[2]

        # Step 2: Auto-rotate portrait → landscape
        # ECG printouts are always wider than tall. A portrait photo means
        # the phone was held vertically — rotate 90° counter-clockwise.
        if h > w:
            image = torch.rot90(image, k=1, dims=(1, 2))
            h, w = image.shape[1], image.shape[2]
            logger.info(
                "Auto-rotated portrait image to landscape: %dx%d",
                w, h,
            )

        if getattr(self, "enable_perspective_correction", False):
            self.last_info.page_correction_version = PERSPECTIVE_CORRECTION_VERSION
            correction = correct_perspective(image)
            image = correction.image
            self.last_info.page_correction_applied = correction.applied
            self.last_info.page_correction_confidence = correction.confidence
            h, w = image.shape[1], image.shape[2]
            if correction.applied:
                logger.info(
                    "Applied conservative perspective correction: %dx%d",
                    w,
                    h,
                )

        # Step 4: Downscale oversized phone photos to prevent memory explosion
        if max(h, w) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            original_dtype = image.dtype
            image = F_torch.interpolate(
                image.unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(original_dtype)
            logger.info(
                "Downscaled large image: %dx%d -> %dx%d (%.2fx) to limit memory",
                w, h, new_w, new_h, scale,
            )
            h, w = new_h, new_w

        if getattr(self, "enable_shadow_normalization", False):
            self.last_info.shadow_normalization_version = SHADOW_NORMALIZATION_VERSION
            normalization = normalize_broad_shadow(image)
            image = normalization.image
            self.last_info.shadow_normalization_applied = normalization.applied
            self.last_info.shadow_score = normalization.shadow_score

        # Step 6: Upscale small images (capped at 2x to avoid excessive blur)
        if min(h, w) < MIN_IMAGE_DIMENSION:
            scale = MIN_IMAGE_DIMENSION / min(h, w)
            scale = min(scale, self._MAX_UPSCALE_FACTOR)
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

    def _run_inference(
        self,
        image_tensor: torch.Tensor,
        layout_hint: str | None = None,
        *,
        apply_dewarping: bool | None = None,
    ) -> dict:
        """Run Open-ECG-Digitizer forward pass.

        Args:
            image_tensor: Preprocessed image tensor.
            layout_hint: Optional substring to filter layout candidates
                (e.g. "3x4" restricts to 3x4 layouts only).
        """
        import time

        start = time.time()
        with self._override_wrapper_settings(apply_dewarping=apply_dewarping):
            result: dict = self._wrapper(
                image_tensor, layout_should_include_substring=layout_hint,
            )
        self._trim_debug_payload(result)
        elapsed = time.time() - start
        logger.info("Digitization inference took %.1f seconds", elapsed)
        return result

    def _run_best_inference(
        self,
        image_tensor: torch.Tensor,
        layout_hint: str | None = None,
    ) -> dict:
        """Retry difficult cases with alternate orientation/dewarping."""
        best_result = self._run_inference(image_tensor, layout_hint=layout_hint)
        best_result["processing_mode"] = "default"
        best_score = self._score_raw_result(best_result)
        best_image_tensor = image_tensor

        if (
            self.enable_orientation_retry
            and self._should_retry_with_orientation(best_result)
        ):
            rotated_image_tensor = torch.rot90(image_tensor, k=2, dims=(2, 3))
            rotated_result = self._run_inference(
                rotated_image_tensor,
                layout_hint=layout_hint,
            )
            rotated_result["processing_mode"] = "rotated_180_retry"
            rotated_score = self._score_raw_result(rotated_result)
            if rotated_score > best_score:
                logger.info(
                    "Selected rotated orientation result (score %.2f > %.2f)",
                    rotated_score,
                    best_score,
                )
                del best_result
                best_result = rotated_result
                best_score = rotated_score
                best_image_tensor = rotated_image_tensor
            else:
                logger.info(
                    "Kept default orientation result (score %.2f >= %.2f)",
                    best_score,
                    rotated_score,
                )
                del rotated_result, rotated_image_tensor

        if self.enable_dewarping_retry and self._should_retry_with_dewarping(best_result):
            # Free memory from first pass before allocating second pass
            gc.collect()
            retry_result = self._run_inference(
                best_image_tensor,
                layout_hint=layout_hint,
                apply_dewarping=True,
            )
            retry_result["processing_mode"] = (
                "rotated_180_dewarped_retry"
                if best_image_tensor is not image_tensor
                else "dewarped_retry"
            )
            retry_score = self._score_raw_result(retry_result)
            if retry_score > best_score:
                logger.info(
                    "Selected dewarped retry result (score %.2f > %.2f)",
                    retry_score,
                    best_score,
                )
                del best_result
                return retry_result
            logger.info(
                "Kept default digitization result (score %.2f >= %.2f)",
                best_score,
                retry_score,
            )
            del retry_result

        return best_result

    @staticmethod
    def _should_retry_with_orientation(result: dict) -> bool:
        """Try the opposite landscape orientation for weak first-pass results."""
        signal_dict = result.get("signal", {})
        layout_cost = float(signal_dict.get("layout_matching_cost", float("inf")))
        raw_lines = signal_dict.get("raw_lines")
        raw_lines_count = 0 if raw_lines is None else int(raw_lines.shape[0])
        detected_count = int(signal_dict.get("n_detected", 0))
        return layout_cost > 1.2 or raw_lines_count < 3 or detected_count < 8

    @staticmethod
    def _trim_debug_payload(result: dict) -> None:
        """Drop bulky tensors we do not use outside the vendor wrapper.

        Keeping full-resolution aligned images and probability maps in the
        Gradio worker increases peak RSS and makes the macOS launch agent
        more likely to be killed under memory pressure.
        """
        result.pop("input_image", None)
        result.pop("aligned", None)
        result.pop("source_points", None)

    def _should_retry_with_dewarping(self, result: dict) -> bool:
        """Retry cases whose geometry suggests a poor extraction."""
        signal_dict = result.get("signal", {})
        layout_cost = float(signal_dict.get("layout_matching_cost", float("inf")))
        raw_lines = signal_dict.get("raw_lines")
        raw_lines_count = 0 if raw_lines is None else int(raw_lines.shape[0])
        detected_count = int(signal_dict.get("n_detected", 0))
        return (
            not getattr(self._wrapper, "apply_dewarping", False)
            and (
                layout_cost > 1.2
                or raw_lines_count < 4
                or detected_count < 10
            )
        )

    def _score_raw_result(self, result: dict) -> float:
        """Score a raw digitization result to pick the cleanest attempt."""
        signal_dict = result.get("signal", {})
        layout_cost = float(signal_dict.get("layout_matching_cost", float("inf")))
        raw_lines = signal_dict.get("raw_lines")
        raw_lines_count = 0 if raw_lines is None else int(raw_lines.shape[0])
        detected_count = int(signal_dict.get("n_detected", 0))
        canonical = signal_dict.get("canonical_lines")

        finite_ratio = 0.0
        nonempty_leads = 0
        if canonical is not None:
            finite_mask = torch.isfinite(canonical)
            finite_ratio = float(finite_mask.float().mean().item())
            nonempty_leads = int(torch.isfinite(canonical).any(dim=1).sum().item())

        return (
            nonempty_leads * 4.0
            + raw_lines_count * 3.0
            + detected_count * 1.5
            + finite_ratio * 5.0
            - layout_cost * 10.0
        )

    @contextmanager
    def _override_wrapper_settings(
        self,
        *,
        apply_dewarping: bool | None = None,
    ):
        """Temporarily override wrapper runtime settings for fallback attempts."""
        original_apply_dewarping = getattr(self._wrapper, "apply_dewarping", None)
        try:
            if apply_dewarping is not None:
                self._wrapper.apply_dewarping = apply_dewarping
            yield
        finally:
            if original_apply_dewarping is not None:
                self._wrapper.apply_dewarping = original_apply_dewarping

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
        if not torch.isfinite(canonical).any():
            raise RuntimeError(
                "Digitizer produced no finite canonical signal values. "
                "Signal extraction likely failed."
            )
        return canonical

    def _validate_digitized_signal(
        self,
        signal: np.ndarray,
        layout_hint: str | None,
    ) -> None:
        """Reject silent digitization failures before downstream inference."""
        issues: list[str] = []
        if self.last_info.raw_lines_count < MIN_REQUIRED_RAW_LINES:
            issues.append(
                f"only {self.last_info.raw_lines_count} raw signal lines "
                f"were extracted (expected >= {MIN_REQUIRED_RAW_LINES})"
            )
        if self.last_info.nonzero_leads_count < MIN_REQUIRED_NONZERO_LEADS:
            issues.append(
                f"only {self.last_info.nonzero_leads_count} leads contain "
                f"usable signal energy (expected >= {MIN_REQUIRED_NONZERO_LEADS})"
            )
        if issues:
            hint_text = ""
            if layout_hint is None:
                hint_text = (
                    " Try providing an explicit layout hint such as "
                    "'3x4+1R' or '6x2'."
                )
            raise RuntimeError(
                "Digitization output failed quality checks: "
                + "; ".join(issues)
                + hint_text
            )

    @staticmethod
    def _count_nonzero_leads(signal: np.ndarray) -> int:
        """Count leads with meaningful activity after normalization."""
        peak_per_lead = np.max(np.abs(signal), axis=1)
        return int(np.sum(peak_per_lead > LEAD_ACTIVITY_THRESHOLD))

    @staticmethod
    def _postprocess(canonical: torch.Tensor) -> np.ndarray:
        """Convert raw canonical_lines to ECGFounder-ready format.

        Steps: expand synchronized paper-layout segments, uV->mV, resample
        to 500Hz/5000pts, highpass 0.5Hz, wavelet denoise, z-score.
        """
        return ECGDigitiser._postprocess_outputs(canonical).model_input

    @staticmethod
    def _postprocess_outputs(canonical: torch.Tensor) -> DigitizedSignals:
        """Return calibrated mV signal plus the normalized model input."""
        signal = canonical.cpu().numpy().astype(np.float64)

        # Ensure exactly 12 leads before expanding the paper layout.
        signal = _pad_or_truncate_leads(signal, NUM_LEADS)

        # Preserve the uncropped full-width rhythm strip BEFORE expansion crops
        # it to a single column. This keeps the genuine 10 s RR sequence for
        # heart-rate analysis (diagnosis still uses the cropped/tiled signal).
        rhythm_canonical = _extract_rhythm_strip(signal)

        # Canonical NaNs encode where each printed lead segment begins and
        # ends. Expand those exact shared column windows before replacing
        # missing values so simultaneous leads retain their relative phase.
        signal = _expand_canonical_segments(signal)

        # Convert microvolts to millivolts
        signal = signal / UV_TO_MV

        # Resample to 500 Hz / 5000 samples
        n_points = signal.shape[1]
        if n_points != TARGET_LENGTH:
            signal = _resample_signal(signal, n_points, TARGET_LENGTH)

        # Pad or truncate time axis to exactly 5000 samples
        signal = _pad_or_truncate_time(signal, TARGET_LENGTH)

        # Remove baseline wander (< 0.5 Hz) from paper curvature and
        # lighting gradients. High-pass only — no upper cutoff, so QRS
        # high-frequency content (40-150 Hz) is preserved. ECGFounder
        # was trained on unfiltered WFDB signals with full 0-250 Hz spectrum.
        signal = highpass_filter(signal, TARGET_SAMPLE_RATE)

        # Remove digitization noise (grid artifacts, trace jitter) while
        # preserving QRS/P/T morphology. Wavelet denoising thresholds
        # only high-frequency detail coefficients (> ~30 Hz), leaving
        # diagnostic waveform content untouched.
        signal = wavelet_denoise(signal, TARGET_SAMPLE_RATE)

        calibrated_millivolts = signal.astype(np.float32)
        model_input = _z_score_normalize(signal).astype(np.float32)

        rhythm_strip = (
            _process_rhythm_strip(rhythm_canonical)
            if rhythm_canonical is not None
            else None
        )
        return DigitizedSignals(
            model_input=model_input,
            calibrated_millivolts=calibrated_millivolts,
            rhythm_strip=rhythm_strip,
        )


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


def _to_numpy(array: Any) -> np.ndarray:
    """Convert a torch tensor or array-like to a detached float32 numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy().astype(np.float32)
    return np.asarray(array, dtype=np.float32)


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


def _expand_canonical_segments(signal: np.ndarray) -> np.ndarray:
    """Expand canonical paper-layout segments while preserving synchronization.

    Open-ECG-Digitizer uses NaNs outside each lead's printed time window.
    Leads in the same paper column share exact time boundaries, which must be
    preserved for cross-lead relationships such as Einthoven's law. Partial
    windows are interpolated within their snapped layout column and tiled to
    the full duration. Full-width rhythm strips are cropped to the standard
    lead's synchronized column before tiling when a multi-column layout exists.
    """
    total_len = signal.shape[1]
    expanded = np.zeros_like(signal)
    if total_len == 0:
        return expanded

    finite_spans: list[int] = []
    finite_indices_by_lead: list[np.ndarray] = []
    for lead in signal:
        finite_indices = np.flatnonzero(np.isfinite(lead))
        finite_indices_by_lead.append(finite_indices)
        if finite_indices.size == 0:
            continue
        span = int(finite_indices[-1] - finite_indices[0] + 1)
        if max(2, int(total_len * 0.1)) <= span < total_len * 0.8:
            finite_spans.append(span)

    if finite_spans:
        typical_span = float(np.median(finite_spans))
        n_columns = int(np.clip(round(total_len / typical_span), 1, 6))
    else:
        n_columns = 1
    column_edges = np.linspace(0, total_len, n_columns + 1, dtype=int)

    for lead_idx, finite_indices in enumerate(finite_indices_by_lead):
        if finite_indices.size == 0:
            continue

        lead = signal[lead_idx]
        span = int(finite_indices[-1] - finite_indices[0] + 1)
        if span >= total_len * 0.8:
            if n_columns == 1:
                expanded[lead_idx] = _interpolate_finite_values(lead)
                continue
            leads_per_column = int(np.ceil(signal.shape[0] / n_columns))
            column_idx = min(n_columns - 1, lead_idx // leads_per_column)
        elif n_columns > 1:
            center = float(finite_indices[0] + finite_indices[-1]) / 2.0
            column_idx = min(n_columns - 1, int(center * n_columns / total_len))
        else:
            start = int(finite_indices[0])
            end = int(finite_indices[-1] + 1)
            segment = lead[start:end]
            segment = _interpolate_finite_values(segment)
            repeats = (total_len + len(segment) - 1) // len(segment)
            expanded[lead_idx] = np.tile(segment, repeats)[:total_len]
            continue

        start = int(column_edges[column_idx])
        end = int(column_edges[column_idx + 1])
        segment = lead[start:end]
        if not np.isfinite(segment).any():
            segment = lead[finite_indices[0] : finite_indices[-1] + 1]
        segment = _interpolate_finite_values(segment)
        repeats = (total_len + len(segment) - 1) // len(segment)
        expanded[lead_idx] = np.tile(segment, repeats)[:total_len]

    return expanded


def _extract_rhythm_strip(signal: np.ndarray) -> np.ndarray | None:
    """Preserve full-width rhythm leads uncropped for heart-rate analysis.

    Paper layouts print a continuous rhythm strip (usually Lead II) spanning the
    full page width alongside short ~2.5 s lead segments. ``_expand_canonical_
    segments`` crops that strip to a single column to keep diagnosis leads
    temporally coherent — which discards the real beat-to-beat (RR) sequence.
    For heart rate we want the opposite: the rhythm lead at full duration.

    Args:
        signal: (n_leads, samples) canonical array with NaNs outside each
            printed lead's time window (pre-expansion).

    Returns:
        A (n_leads, samples) array with full-width leads interpolated over the
        whole window and every other lead zeroed, or None when no full-width
        rhythm strip exists (e.g. a pure-column 12x1 layout).
    """
    total_len = signal.shape[1]
    if total_len == 0:
        return None

    strip = np.zeros_like(signal)
    found = False
    for lead_idx, lead in enumerate(signal):
        finite_indices = np.flatnonzero(np.isfinite(lead))
        if finite_indices.size == 0:
            continue
        span = int(finite_indices[-1] - finite_indices[0] + 1)
        if span >= total_len * 0.8:
            strip[lead_idx] = _interpolate_finite_values(lead)
            found = True

    return strip if found else None


def _process_rhythm_strip(rhythm_canonical: np.ndarray) -> np.ndarray:
    """Resample, filter and normalize the uncropped rhythm strip for HR.

    Mirrors the model-input chain (uV->mV, 500 Hz/5000 pts, highpass, wavelet,
    z-score) but skips the segment expansion that crops the strip, so the
    genuine RR timing survives.
    """
    signal = rhythm_canonical / UV_TO_MV

    n_points = signal.shape[1]
    if n_points != TARGET_LENGTH:
        signal = _resample_signal(signal, n_points, TARGET_LENGTH)
    signal = _pad_or_truncate_time(signal, TARGET_LENGTH)

    signal = highpass_filter(signal, TARGET_SAMPLE_RATE)
    signal = wavelet_denoise(signal, TARGET_SAMPLE_RATE)
    return _z_score_normalize(signal).astype(np.float32)


def _interpolate_finite_values(values: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs, extending edge values over missing boundaries."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    if finite.all():
        return values.copy()

    positions = np.arange(len(values))
    interpolated: np.ndarray = np.interp(
        positions,
        positions[finite],
        values[finite],
    )
    return interpolated


def _bandpass_filter(
    signal: np.ndarray,
    sample_rate: int,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
) -> np.ndarray:
    """Remove grid artifacts and baseline wander from digitized ECG.

    Paper ECG photos contain residual grid line patterns (typically at
    frequencies above 40 Hz) and baseline drift (below 0.5 Hz). Both
    cause false positive diagnoses in ECGFounder, which was trained on
    clean WFDB recordings. The 0.5-40 Hz band preserves diagnostic
    ECG morphology (P, QRS, T, ST) while removing digitization noise.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = sample_rate / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = butter(N=3, Wn=[low, high], btype="bandpass", output="sos")
    filtered = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        lead = signal[i]
        if np.std(lead) < 1e-6:
            filtered[i] = lead
            continue
        filtered[i] = sosfiltfilt(sos, lead)
    return filtered


def _z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """Global z-score normalization (zero mean, unit variance)."""
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8)
