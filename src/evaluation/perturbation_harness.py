"""Bounded, grid-preserving image perturbations for reconstruction stability.

A stable digitization should survive a small, clinically irrelevant geometric
nudge of the source photo. Fragile extractions (layout misdetection, lead
mis-assignment) tend to flip under such a nudge, so comparing two
reconstructions across a bounded perturbation is a candidate reference-free
fidelity signal. See `evaluate_reconstruction_pair` for the comparison side.

Translations are integer-pixel shifts with edge replication — interpolation
free, so the ECG grid is shifted but otherwise untouched. Scales are a single
center-anchored bilinear resample; grid spacing changes uniformly and stays a
valid grid. Both stay inside conservative bounds enforced at construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# Conservative bounds: clinically irrelevant, grid-preserving. A 5% scale or a
# 20 px shift is far below the spacing the digitizer relies on, yet enough to
# perturb a borderline layout decision.
MIN_SCALE: float = 0.95
MAX_SCALE: float = 1.05
MAX_TRANSLATION_PX: int = 20


@dataclass(frozen=True)
class PerturbationSpec:
    """A bounded geometric perturbation applied about the image center."""

    label: str
    scale: float = 1.0
    translate_x: int = 0
    translate_y: int = 0

    def validate(self) -> None:
        """Reject perturbations outside the clinically irrelevant envelope."""
        if not MIN_SCALE <= self.scale <= MAX_SCALE:
            raise ValueError(
                f"scale {self.scale} outside [{MIN_SCALE}, {MAX_SCALE}]"
            )
        if abs(self.translate_x) > MAX_TRANSLATION_PX or abs(self.translate_y) > MAX_TRANSLATION_PX:
            raise ValueError(
                f"translation ({self.translate_x}, {self.translate_y}) exceeds "
                f"+/-{MAX_TRANSLATION_PX} px"
            )

    @property
    def is_identity(self) -> bool:
        """Whether the spec leaves the image unchanged."""
        return self.scale == 1.0 and self.translate_x == 0 and self.translate_y == 0


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8 RGB")


def apply_scale(image: np.ndarray, scale: float) -> np.ndarray:
    """Center-zoom the image by `scale`, preserving the original frame size.

    Zoom-in (scale > 1) resamples up then center-crops; zoom-out resamples down
    then edge-pads. A single bilinear resample keeps interpolation artefacts
    minimal while changing grid spacing uniformly.
    """
    if scale == 1.0:
        return image.copy()
    height, width = image.shape[:2]
    scaled_height = max(1, round(height * scale))
    scaled_width = max(1, round(width * scale))
    resized = np.asarray(
        Image.fromarray(image).resize(
            (scaled_width, scaled_height), resample=Image.Resampling.BILINEAR
        )
    )
    output = np.empty_like(image)
    if scale > 1.0:
        top = (scaled_height - height) // 2
        left = (scaled_width - width) // 2
        output[:] = resized[top : top + height, left : left + width]
    else:
        # Edge-replicate the border exposed by shrinking, anchored at center.
        top = (height - scaled_height) // 2
        left = (width - scaled_width) // 2
        padded = np.pad(
            resized,
            ((top, height - scaled_height - top), (left, width - scaled_width - left), (0, 0)),
            mode="edge",
        )
        output[:] = padded
    return output


def apply_translation(image: np.ndarray, translate_x: int, translate_y: int) -> np.ndarray:
    """Shift the image by integer pixels, replicating the exposed edge.

    Interpolation free: every retained pixel keeps its exact value, so the grid
    is shifted but never blurred.
    """
    if translate_x == 0 and translate_y == 0:
        return image.copy()
    height, width = image.shape[:2]
    # Pad on the side the content moves away from, then crop the far side.
    pad_left = max(translate_x, 0)
    pad_right = max(-translate_x, 0)
    pad_top = max(translate_y, 0)
    pad_bottom = max(-translate_y, 0)
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    return padded[pad_bottom : pad_bottom + height, pad_right : pad_right + width].copy()


def apply_perturbation(image: np.ndarray, spec: PerturbationSpec) -> np.ndarray:
    """Apply a validated bounded perturbation, returning a same-shape uint8 array."""
    _validate_image(image)
    spec.validate()
    if spec.is_identity:
        return image.copy()
    scaled = apply_scale(image, spec.scale)
    return apply_translation(scaled, spec.translate_x, spec.translate_y)


def default_perturbation_set() -> tuple[PerturbationSpec, ...]:
    """A small bounded probe set: pure shift, pure zoom, and a combined nudge.

    Worst-case disagreement across these probes is the candidate stability
    signal. Kept tiny so the tune pilot stays affordable on CPU inference.
    """
    return (
        PerturbationSpec(label="translate_+12_-8", translate_x=12, translate_y=-8),
        PerturbationSpec(label="scale_1.03", scale=1.03),
        PerturbationSpec(
            label="scale_0.97_translate_-10_+10",
            scale=0.97,
            translate_x=-10,
            translate_y=10,
        ),
    )


__all__ = [
    "MIN_SCALE",
    "MAX_SCALE",
    "MAX_TRANSLATION_PX",
    "PerturbationSpec",
    "apply_scale",
    "apply_translation",
    "apply_perturbation",
    "default_perturbation_set",
]
