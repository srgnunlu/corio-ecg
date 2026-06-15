"""Conservatively rectify a high-confidence quadrilateral ECG page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import torch

MIN_PAGE_AREA_RATIO = 0.45
MAX_PAGE_AREA_RATIO = 0.92
MIN_LANDSCAPE_ASPECT = 1.20
MAX_LANDSCAPE_ASPECT = 2.40
MIN_DISTORTION_RATIO = 0.025
MAX_DISTORTION_RATIO = 0.25
MIN_CONFIDENCE = 0.80
DETECTION_MAX_DIMENSION = 1400
PERSPECTIVE_CORRECTION_VERSION = "conservative-perspective-v1"


@dataclass(frozen=True)
class PerspectiveCorrectionResult:
    """Result of an optional conservative page rectification."""

    image: torch.Tensor
    applied: bool
    confidence: float


def _order_corners(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return cast(
        np.ndarray,
        np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
        ),
    )


def _dimensions(corners: np.ndarray) -> tuple[float, float]:
    top_left, top_right, bottom_right, bottom_left = corners
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    return float(width), float(height)


def _candidate_score(
    corners: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float] | None:
    area = abs(float(cv2.contourArea(corners)))
    area_ratio = area / (image_width * image_height)
    width, height = _dimensions(corners)
    aspect_ratio = width / max(height, 1.0)
    if not (
        MIN_PAGE_AREA_RATIO <= area_ratio <= MAX_PAGE_AREA_RATIO
        and MIN_LANDSCAPE_ASPECT <= aspect_ratio <= MAX_LANDSCAPE_ASPECT
    ):
        return None

    x, y, bound_width, bound_height = cv2.boundingRect(corners.astype(np.int32))
    rectangle = np.asarray(
        [
            [x, y],
            [x + bound_width, y],
            [x + bound_width, y + bound_height],
            [x, y + bound_height],
        ],
        dtype=np.float32,
    )
    distortion = float(
        np.mean(np.linalg.norm(corners - rectangle, axis=1))
        / np.hypot(image_width, image_height)
    )
    if not MIN_DISTORTION_RATIO <= distortion <= MAX_DISTORTION_RATIO:
        return None
    confidence = min(1.0, area_ratio + distortion * 7.0)
    return confidence, distortion


def _find_page_corners(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    height, width = image.shape[:2]
    scale = min(1.0, DETECTION_MAX_DIMENSION / max(height, width))
    detection = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    gray = cv2.cvtColor(detection, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[np.ndarray, float] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        corners = _order_corners(polygon.reshape(4, 2).astype(np.float32))
        scored = _candidate_score(
            corners,
            image_width=detection.shape[1],
            image_height=detection.shape[0],
        )
        if scored is None:
            continue
        confidence, _ = scored
        if best is None or confidence > best[1]:
            best = (corners / scale, confidence)
    return best


def correct_perspective(image: torch.Tensor) -> PerspectiveCorrectionResult:
    """Rectify only a high-confidence, meaningfully distorted landscape page."""
    if image.ndim != 3 or image.shape[0] != 3:
        return PerspectiveCorrectionResult(image=image, applied=False, confidence=0.0)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    candidate = _find_page_corners(image_np)
    if candidate is None or candidate[1] < MIN_CONFIDENCE:
        return PerspectiveCorrectionResult(image=image, applied=False, confidence=0.0)

    corners, confidence = candidate
    width, height = _dimensions(corners)
    output_width, output_height = max(1, round(width)), max(1, round(height))
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    corrected = cv2.warpPerspective(
        image_np,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    tensor = torch.from_numpy(corrected).permute(2, 0, 1).to(image.dtype).contiguous()
    return PerspectiveCorrectionResult(image=tensor, applied=True, confidence=confidence)
