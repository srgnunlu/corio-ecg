"""Tests for bounded, grid-preserving image perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.perturbation_harness import (
    MAX_TRANSLATION_PX,
    PerturbationSpec,
    apply_perturbation,
    apply_scale,
    apply_translation,
    default_perturbation_set,
)


def _gradient_image(height: int = 64, width: int = 96) -> np.ndarray:
    rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    plane = ((rows.astype(int) + cols.astype(int)) // 2).astype(np.uint8)
    return np.stack([plane, plane, plane], axis=2)


def _vertical_grid(height: int = 64, width: int = 96, period: int = 8) -> np.ndarray:
    cols = ((np.arange(width) % period) < period // 2).astype(np.uint8) * 255
    plane = np.broadcast_to(cols[None, :], (height, width)).astype(np.uint8)
    return np.stack([plane, plane, plane], axis=2)


def test_identity_spec_returns_unchanged_image() -> None:
    image = _gradient_image()

    result = apply_perturbation(image, PerturbationSpec(label="identity"))

    assert np.array_equal(result, image)


def test_perturbation_preserves_shape_and_dtype() -> None:
    image = _gradient_image()
    for spec in default_perturbation_set():
        result = apply_perturbation(image, spec)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_translation_is_interpolation_free() -> None:
    image = _gradient_image()

    shifted = apply_translation(image, 10, 0)

    # Interior pixels keep exact original values, merely relocated by the shift.
    assert np.array_equal(shifted[:, 10:], image[:, :-10])
    assert set(np.unique(shifted)).issubset(set(np.unique(image)))


def test_pure_translation_preserves_grid_period() -> None:
    grid = _vertical_grid(period=8)

    # Shift by exactly one period: the grid maps onto itself away from edges.
    shifted = apply_translation(grid, 8, 0)

    assert np.array_equal(shifted[:, 8:], grid[:, 8:])


def test_scale_keeps_frame_size_and_value_range() -> None:
    image = _gradient_image()

    zoomed_in = apply_scale(image, 1.04)
    zoomed_out = apply_scale(image, 0.96)

    assert zoomed_in.shape == image.shape
    assert zoomed_out.shape == image.shape
    assert zoomed_in.max() <= 255 and zoomed_out.min() >= 0


def test_out_of_bounds_specs_are_rejected() -> None:
    image = _gradient_image()

    with pytest.raises(ValueError, match="scale"):
        apply_perturbation(image, PerturbationSpec(label="too-big", scale=1.2))
    with pytest.raises(ValueError, match="translation"):
        apply_perturbation(
            image,
            PerturbationSpec(label="too-far", translate_x=MAX_TRANSLATION_PX + 1),
        )


def test_non_rgb_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="height, width, 3"):
        apply_perturbation(np.zeros((4, 4), dtype=np.uint8), PerturbationSpec(label="x"))
    with pytest.raises(ValueError, match="uint8"):
        apply_perturbation(
            np.zeros((4, 4, 3), dtype=np.float32), PerturbationSpec(label="x")
        )
