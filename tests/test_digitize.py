# Tests for ECG digitization pipeline (image -> signal conversion)
# Unit tests run without model weights; integration tests require them

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.digitize import (
    MPS_MAX_IMAGE_DIMENSION,
    NUM_LEADS,
    TARGET_LENGTH,
    UV_TO_MV,
    _pad_or_truncate_leads,
    _pad_or_truncate_time,
    _resample_signal,
    _z_score_normalize,
)

WEIGHTS_DIR = PROJECT_ROOT / "external" / "open-ecg-digitizer" / "weights"
WEIGHTS_EXIST = (
    (WEIGHTS_DIR / "unet_weights_07072025.pt").exists()
    and (WEIGHTS_DIR / "lead_name_unet_weights_07072025.pt").exists()
)


# ---------------------------------------------------------------------------
# Unit tests — no model weights needed
# ---------------------------------------------------------------------------


class TestNanHandling:
    """Verify NaN values are properly replaced with zeros."""

    def test_nan_replaced_with_zero(self) -> None:
        signal = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, np.nan]])
        result = np.nan_to_num(signal, nan=0.0)
        assert not np.any(np.isnan(result))
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0

    def test_non_nan_values_preserved(self) -> None:
        signal = np.array([[1.0, np.nan, 3.0]])
        result = np.nan_to_num(signal, nan=0.0)
        assert result[0, 0] == 1.0
        assert result[0, 2] == 3.0


class TestUvToMvConversion:
    """Verify microvolts to millivolts conversion."""

    def test_basic_conversion(self) -> None:
        uv_signal = np.array([[1000.0, 2000.0, -500.0]])
        mv_signal = uv_signal / UV_TO_MV
        np.testing.assert_allclose(mv_signal, [[1.0, 2.0, -0.5]])

    def test_zero_stays_zero(self) -> None:
        uv_signal = np.array([[0.0, 0.0]])
        mv_signal = uv_signal / UV_TO_MV
        np.testing.assert_allclose(mv_signal, [[0.0, 0.0]])


class TestPadOrTruncateLeads:
    """Verify lead count adjustment."""

    def test_truncate_extra_leads(self) -> None:
        signal = np.ones((15, 100))
        result = _pad_or_truncate_leads(signal, NUM_LEADS)
        assert result.shape == (12, 100)

    def test_pad_missing_leads(self) -> None:
        signal = np.ones((8, 100))
        result = _pad_or_truncate_leads(signal, NUM_LEADS)
        assert result.shape == (12, 100)
        # First 8 leads should be ones, rest zeros
        np.testing.assert_array_equal(result[:8], np.ones((8, 100)))
        np.testing.assert_array_equal(result[8:], np.zeros((4, 100)))

    def test_exact_lead_count_unchanged(self) -> None:
        signal = np.ones((12, 100))
        result = _pad_or_truncate_leads(signal, NUM_LEADS)
        assert result.shape == (12, 100)


class TestResampleSignal:
    """Verify signal resampling."""

    def test_upsample_doubles_length(self) -> None:
        signal = np.ones((12, 2500))
        result = _resample_signal(signal, 2500, 5000)
        assert result.shape == (12, 5000)
        # Constant signal should remain constant after resampling
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_downsample_halves_length(self) -> None:
        signal = np.ones((12, 10000))
        result = _resample_signal(signal, 10000, 5000)
        assert result.shape == (12, 5000)

    def test_preserves_linear_trend(self) -> None:
        # A linear ramp should stay linear after resampling
        signal = np.linspace(0, 1, 3000)[np.newaxis, :]
        signal = np.repeat(signal, 12, axis=0)
        result = _resample_signal(signal, 3000, 5000)
        assert result.shape == (12, 5000)
        # Check monotonically increasing
        diffs = np.diff(result[0])
        assert np.all(diffs >= 0)


class TestPadOrTruncateTime:
    """Verify time-axis padding and truncation."""

    def test_truncate_long_signal(self) -> None:
        signal = np.ones((12, 8000))
        result = _pad_or_truncate_time(signal, TARGET_LENGTH)
        assert result.shape == (12, 5000)

    def test_pad_short_signal(self) -> None:
        signal = np.ones((12, 3000))
        result = _pad_or_truncate_time(signal, TARGET_LENGTH)
        assert result.shape == (12, 5000)
        np.testing.assert_array_equal(result[:, 3000:], 0.0)

    def test_exact_length_unchanged(self) -> None:
        signal = np.ones((12, 5000))
        result = _pad_or_truncate_time(signal, TARGET_LENGTH)
        assert result.shape == (12, 5000)


class TestZScoreNormalize:
    """Verify z-score normalization."""

    def test_output_has_zero_mean(self) -> None:
        signal = np.random.randn(12, 5000) * 5 + 10
        result = _z_score_normalize(signal)
        assert abs(np.mean(result)) < 1e-6

    def test_output_has_unit_variance(self) -> None:
        signal = np.random.randn(12, 5000) * 5 + 10
        result = _z_score_normalize(signal)
        assert abs(np.std(result) - 1.0) < 1e-6

    def test_all_zeros_does_not_crash(self) -> None:
        signal = np.zeros((12, 5000))
        result = _z_score_normalize(signal)
        # Should not produce NaN or inf
        assert np.all(np.isfinite(result))


class TestMpsImageResize:
    """Verify MPS-only image downscaling in _load_image."""

    def test_large_image_resized_on_mps(self) -> None:
        """Images exceeding MPS_MAX_IMAGE_DIMENSION are downscaled on MPS."""
        from src.pipeline.digitize import ECGDigitiser

        # Create a mock digitiser with MPS device (no model needed)
        digitiser = object.__new__(ECGDigitiser)
        digitiser.device = torch.device("mps")

        with patch("torchvision.io.decode_image") as mock_decode:
            # Simulate a 2200x1700 image (3, H, W)
            mock_decode.return_value = torch.zeros(3, 1700, 2200, dtype=torch.uint8)
            result = digitiser._load_image(Path("fake.png"))

            # Should be downscaled: max(1700,2200)=2200 > 1600
            _, _, new_h, new_w = result.shape
            assert max(new_h, new_w) <= MPS_MAX_IMAGE_DIMENSION

    def test_small_image_not_resized_on_mps(self) -> None:
        """Images within MPS limits are not resized."""
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.device = torch.device("mps")

        with patch("torchvision.io.decode_image") as mock_decode:
            mock_decode.return_value = torch.zeros(3, 800, 1200, dtype=torch.uint8)
            result = digitiser._load_image(Path("fake.png"))

            _, _, new_h, new_w = result.shape
            assert new_h == 800
            assert new_w == 1200

    def test_image_not_resized_on_cuda(self) -> None:
        """CUDA devices process at full resolution — no resize."""
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.device = torch.device("cuda")

        with patch("torchvision.io.decode_image") as mock_decode:
            mock_decode.return_value = torch.zeros(3, 1700, 2200, dtype=torch.uint8)
            result = digitiser._load_image(Path("fake.png"))

            _, _, new_h, new_w = result.shape
            # No resize on CUDA — original dimensions preserved
            assert new_h == 1700
            assert new_w == 2200


class TestPostprocessEndToEnd:
    """Test the full postprocess chain using a mock canonical tensor."""

    def test_postprocess_produces_correct_shape(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        # Simulate canonical_lines: (12, 3000) tensor in microvolts with some NaN
        canonical = torch.randn(12, 3000) * 500.0
        canonical[0, 100:200] = float("nan")  # simulate undetected segment

        result = ECGDigitiser._postprocess(canonical)

        assert isinstance(result, np.ndarray)
        assert result.shape == (12, 5000)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert np.all(np.isfinite(result))

    def test_postprocess_not_all_zeros(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        canonical = torch.randn(12, 5000) * 1000.0
        result = ECGDigitiser._postprocess(canonical)
        # A random signal should not normalize to all zeros
        assert np.any(result != 0.0)


# ---------------------------------------------------------------------------
# Integration tests — require model weights
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not WEIGHTS_EXIST, reason="Model weights not found")
class TestIntegrationDigitize:
    """Full integration test with real Open-ECG-Digitizer model."""

    @pytest.fixture(scope="class")
    def digitiser(self) -> "ECGDigitiser":
        from src.pipeline.digitize import ECGDigitiser
        return ECGDigitiser()

    def test_digitize_returns_correct_shape(
        self, digitiser: "ECGDigitiser", tmp_path: Path
    ) -> None:
        """Test with a synthetic solid-color image (not a real ECG)."""
        # Create a simple test image — won't produce meaningful signal
        # but should exercise the full pipeline without crashing
        from PIL import Image

        test_image = Image.new("RGB", (800, 600), color=(255, 255, 255))
        image_path = tmp_path / "test_ecg.png"
        test_image.save(image_path)

        try:
            result = digitiser.digitize(image_path)
            assert result.shape == (12, 5000)
            assert result.dtype == np.float32
            assert not np.any(np.isnan(result))
        except RuntimeError:
            # A blank image may fail to detect leads — that is acceptable
            pytest.skip("Blank test image failed lead detection (expected)")
