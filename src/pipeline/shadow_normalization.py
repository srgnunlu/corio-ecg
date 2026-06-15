"""Conservatively normalize broad illumination shadows on ECG pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import torch

SHADOW_NORMALIZATION_VERSION = "conservative-shadow-v1"
MIN_BRIGHT_PERCENTILE = 150.0
MIN_SHADOW_SCORE = 0.20
MAX_DETECTION_DIMENSION = 1000
NORMALIZATION_STRENGTH = 0.65


@dataclass(frozen=True)
class ShadowNormalizationResult:
    """Result of optional broad-shadow normalization."""

    image: torch.Tensor
    applied: bool
    shadow_score: float


def _illumination_map(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    scale = min(1.0, MAX_DETECTION_DIMENSION / max(height, width))
    detection = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    kernel = max(31, round(min(detection.shape) / 5))
    if kernel % 2 == 0:
        kernel += 1
    illumination = cv2.GaussianBlur(detection, (kernel, kernel), 0)
    if illumination.shape != gray.shape:
        illumination = cv2.resize(
            illumination,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    return cast(np.ndarray, illumination.astype(np.float32))


def normalize_broad_shadow(image: torch.Tensor) -> ShadowNormalizationResult:
    """Normalize only a bright page with a large smooth luminance gradient."""
    if image.ndim != 3 or image.shape[0] != 3:
        return ShadowNormalizationResult(image=image, applied=False, shadow_score=0.0)

    image_np = image.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    illumination = _illumination_map(gray)
    low, high = np.percentile(illumination, [10, 90])
    shadow_score = float((high - low) / max(high, 1.0))
    if high < MIN_BRIGHT_PERCENTILE or shadow_score < MIN_SHADOW_SCORE:
        return ShadowNormalizationResult(
            image=image,
            applied=False,
            shadow_score=shadow_score,
        )

    target = float(high)
    gain = np.clip(target / np.maximum(illumination, 1.0), 0.75, 1.50)
    gain = 1.0 + (gain - 1.0) * NORMALIZATION_STRENGTH
    normalized = np.clip(image_np.astype(np.float32) * gain[:, :, None], 0, 255)
    tensor = (
        torch.from_numpy(normalized.astype(np.uint8))
        .permute(2, 0, 1)
        .to(image.dtype)
        .contiguous()
    )
    return ShadowNormalizationResult(
        image=tensor,
        applied=True,
        shadow_score=shadow_score,
    )
