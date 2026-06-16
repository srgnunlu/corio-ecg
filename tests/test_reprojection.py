# Tests for reference-free image-space re-projection fidelity.

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.reprojection import reprojection_fidelity

HEIGHT = 120
WIDTH = 200


def _ink_band(centerline: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Build an (H, W) probability map with a thick ink band along centerline."""
    prob = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    columns = np.arange(WIDTH)
    for offset in range(-thickness, thickness + 1):
        rows = np.clip(centerline + offset, 0, HEIGHT - 1)
        prob[rows, columns] = 1.0
    return prob


def _sine_centerline(center: int = 60, amplitude: int = 15) -> np.ndarray:
    columns = np.arange(WIDTH)
    return np.rint(center + amplitude * np.sin(2 * np.pi * columns / 50)).astype(int)


class TestReprojectionFidelity:
    """Verify the overlay metric rewards on-ink traces and punishes failures."""

    def test_faithful_overlay_scores_high(self) -> None:
        """A trace that follows the ink centroid overlaps almost perfectly."""
        centerline = _sine_centerline()
        prob = _ink_band(centerline)
        raw_lines = centerline.astype(np.float64)[None, :]

        result = reprojection_fidelity(prob, raw_lines)

        assert result.precision > 0.95
        assert result.recall > 0.95
        assert result.residual < 0.05

    def test_off_ink_reconstruction_low_precision(self) -> None:
        """A trace far from the ink does not sit on it."""
        centerline = _sine_centerline()
        prob = _ink_band(centerline)
        off = np.clip(centerline + 40, 0, HEIGHT - 1).astype(np.float64)[None, :]

        result = reprojection_fidelity(prob, off)

        assert result.precision < 0.1
        assert result.residual > 0.8

    def test_missed_lead_low_recall(self) -> None:
        """Two ink bands but only one reconstructed trace leaves ink unexplained."""
        band_a = np.full(WIDTH, 30)
        band_b = np.full(WIDTH, 90)
        prob = _ink_band(band_a) + _ink_band(band_b)
        prob = np.clip(prob, 0.0, 1.0)
        raw_lines = band_a.astype(np.float64)[None, :]

        result = reprojection_fidelity(prob, raw_lines)

        assert result.precision > 0.9  # the one trace sits on its band
        assert result.recall < 0.6  # the other band is unexplained
        assert 0.3 < result.residual < 0.6

    def test_crop_offset_alignment(self) -> None:
        """A cropped raw_lines aligns to the ink frame via extraction_crop_x0."""
        centerline = _sine_centerline()
        prob = _ink_band(centerline)
        crop = 30
        raw_lines = centerline[crop:].astype(np.float64)[None, :]

        aligned = reprojection_fidelity(prob, raw_lines, extraction_crop_x0=crop)
        misaligned = reprojection_fidelity(prob, raw_lines, extraction_crop_x0=0)

        assert aligned.precision > 0.95
        assert aligned.precision > misaligned.precision

    def test_empty_traces_max_residual(self) -> None:
        """No reconstructed traces yields residual 1.0 without crashing."""
        prob = _ink_band(_sine_centerline())
        empty = np.empty((0, WIDTH), dtype=np.float64)

        result = reprojection_fidelity(prob, empty)

        assert result.residual == 1.0
        assert result.reconstructed_pixels == 0

    def test_no_ink_max_residual(self) -> None:
        """A blank probability map yields residual 1.0."""
        prob = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        raw_lines = _sine_centerline().astype(np.float64)[None, :]

        result = reprojection_fidelity(prob, raw_lines)

        assert result.residual == 1.0
        assert result.ink_pixels == 0

    def test_nan_gaps_are_skipped(self) -> None:
        """NaN columns (no trace there) must not raise or render garbage."""
        centerline = _sine_centerline()
        prob = _ink_band(centerline)
        raw_lines = centerline.astype(np.float64)[None, :].copy()
        raw_lines[0, 50:80] = np.nan

        result = reprojection_fidelity(prob, raw_lines)

        assert result.precision > 0.95  # remaining trace still on ink
        assert result.reconstructed_pixels == WIDTH - 30

    def test_requires_2d_probability(self) -> None:
        """A non-2-D probability map is a programming error."""
        with pytest.raises(ValueError, match="must be 2-D"):
            reprojection_fidelity(np.zeros(WIDTH, dtype=np.float32), np.zeros((1, WIDTH)))
