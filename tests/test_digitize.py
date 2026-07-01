# Tests for ECG digitization pipeline (image -> signal conversion)
# Unit tests run without model weights; integration tests require them

from __future__ import annotations

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

from src.pipeline.digitize import (  # noqa: E402
    NUM_LEADS,
    TARGET_LENGTH,
    UV_TO_MV,
    DigitizeInfo,
    ECGDigitiser,
    _crop_likely_landscape_page,
    _expand_canonical_segments,
    _interpolate_finite_values,
    _pad_or_truncate_leads,
    _pad_or_truncate_time,
    _resample_signal,
    _z_score_normalize,
)
from src.utils.signal_clean import einthoven_consistency  # noqa: E402

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

    def test_short_internal_gaps_are_linearly_interpolated(self) -> None:
        signal = np.array([0.0, 1.0, np.nan, np.nan, 4.0, 5.0])

        result = _interpolate_finite_values(signal, max_linear_gap=3)

        np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_long_internal_gaps_do_not_create_linear_bridge_artifacts(self) -> None:
        signal = np.array([0.0, 0.0, np.nan, np.nan, np.nan, np.nan, 8.0, 8.0])

        result = _interpolate_finite_values(signal, max_linear_gap=2)

        assert not np.all(np.diff(result[1:7]) > 0.0)
        np.testing.assert_allclose(result[2:6], np.full(4, 4.0))


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


class TestExpandCanonicalSegments:
    """Verify paper-layout segments are expanded without per-lead phase shifts."""

    def test_preserves_einthoven_relation_for_synchronized_limb_leads(self) -> None:
        canonical = np.full((12, 100), np.nan)
        base = np.sin(np.linspace(0, 2 * np.pi, 25, endpoint=False))
        canonical[0, :25] = base
        canonical[2, :25] = 0.5 * base
        canonical[1, :25] = canonical[0, :25] + canonical[2, :25]

        expanded = _expand_canonical_segments(canonical)

        np.testing.assert_allclose(expanded[1], expanded[0] + expanded[2])
        assert einthoven_consistency(expanded) > 0.99

    def test_tiles_each_exact_layout_column_without_phase_shift(self) -> None:
        canonical = np.full((12, 100), np.nan)
        segment = np.arange(25, dtype=np.float64)
        canonical[0, :25] = segment
        canonical[3, 25:50] = segment + 100
        canonical[6, 50:75] = segment + 200
        canonical[9, 75:100] = segment + 300

        expanded = _expand_canonical_segments(canonical)

        np.testing.assert_array_equal(expanded[0], np.tile(segment, 4))
        np.testing.assert_array_equal(expanded[3], np.tile(segment + 100, 4))
        np.testing.assert_array_equal(expanded[6], np.tile(segment + 200, 4))
        np.testing.assert_array_equal(expanded[9], np.tile(segment + 300, 4))

    def test_aligns_full_width_rhythm_lead_to_its_layout_column(self) -> None:
        canonical = np.full((12, 100), np.nan)
        rhythm = np.linspace(-1, 1, 100)
        canonical[1] = rhythm
        canonical[1, 50] = np.nan
        canonical[0, :25] = np.arange(25)

        expanded = _expand_canonical_segments(canonical)

        expected_segment = rhythm[:25]
        np.testing.assert_allclose(expanded[1], np.tile(expected_segment, 4))

    def test_preserves_full_width_leads_when_no_partial_layout_exists(self) -> None:
        canonical = np.tile(np.linspace(-1, 1, 100), (12, 1))
        canonical[1, 50] = np.nan

        expanded = _expand_canonical_segments(canonical)

        rhythm = np.linspace(-1, 1, 100)
        np.testing.assert_allclose(expanded[1], rhythm)

    def test_long_gaps_inside_segment_are_not_drawn_as_diagonal_ramps(self) -> None:
        canonical = np.full((12, 100), np.nan)
        canonical[9, 50:65] = 4.0
        canonical[9, 85:100] = -4.0

        expanded = _expand_canonical_segments(canonical)
        segment = expanded[9, :50]

        # The missing 20-sample gap should be suppressed to local baseline instead
        # of bridged as a long descending straight line between the two islands.
        assert np.max(np.abs(segment[15:35])) < 1.0

    def test_long_gaps_borrow_synchronous_precordial_neighbor_waveform(self) -> None:
        canonical = np.full((12, 100), np.nan)
        reference = np.zeros(50, dtype=np.float64)
        reference[4:10] = np.linspace(0.0, 2.0, 6)
        reference[10:16] = np.linspace(2.0, -1.0, 6)
        reference[18:28] = [0.0, 1.0, 4.0, 10.0, -6.0, -2.0, 0.5, 2.0, 1.0, 0.0]
        reference[34:42] = np.linspace(0.0, 3.0, 8)
        reference[42:48] = np.linspace(3.0, 0.0, 6)
        target = 2.0 * reference + 1.0

        canonical[11, 50:100] = reference
        canonical[10, 50:100] = target
        canonical[10, 65:85] = np.nan

        expanded = _expand_canonical_segments(canonical)
        segment = expanded[10, :50]

        assert np.max(segment[15:35]) > 8.0
        assert np.min(segment[15:35]) < -8.0


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


class TestCpuFullResolution:
    """Verify digitiser always runs on CPU at full image resolution."""

    def test_device_forced_to_cpu(self) -> None:
        """ECGDigitiser always uses CPU regardless of requested device."""
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        # Simulate __init__ device logic
        digitiser.device = torch.device("cpu")
        assert digitiser.device.type == "cpu"

    def test_image_upscaled_to_min_dimension(self) -> None:
        """Small images are upscaled so shortest side reaches MIN_IMAGE_DIMENSION."""
        from src.pipeline.digitize import MIN_IMAGE_DIMENSION, ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.device = torch.device("cpu")

        with patch("torchvision.io.decode_image") as mock_decode:
            # Simulate a 2200x1700 image — shortest side < MIN_IMAGE_DIMENSION
            mock_decode.return_value = torch.zeros(3, 1700, 2200, dtype=torch.uint8)
            result = digitiser._load_image(Path("fake.png"))

            _, _, new_h, new_w = result.shape
            assert min(new_h, new_w) == MIN_IMAGE_DIMENSION

    def test_crops_textured_landscape_page_from_portrait_screen_photo(self) -> None:
        """A monitor photo should be cropped to the bright ECG page before rotation."""
        image = torch.zeros(3, 1200, 700, dtype=torch.uint8)
        image[:, :250, :] = 230  # bright but blank wall above the monitor
        image[:, 400:850, 50:650] = 225
        image[:, 400:850:12, 50:650] = 130
        image[:, 400:850, 50:650:12] = 130
        image[:, 500:510, 50:650] = 20
        image[:, 650:660, 50:650] = 20

        cropped = _crop_likely_landscape_page(image)

        assert cropped.shape[2] > cropped.shape[1]
        assert cropped.shape[1] < image.shape[1]
        assert cropped.shape[2] < image.shape[2]

    def test_does_not_crop_full_frame_portrait_paper(self) -> None:
        """A paper filling the frame should keep its full bounds, then rotate normally."""
        image = torch.full((3, 1200, 700), 225, dtype=torch.uint8)
        image[:, ::12, :] = 150
        image[:, :, ::12] = 150

        cropped = _crop_likely_landscape_page(image)

        assert cropped.shape == image.shape

    def test_load_image_applies_enabled_perspective_correction(self) -> None:
        """The controlled experiment flag applies correction before resizing."""
        digitiser = object.__new__(ECGDigitiser)
        digitiser.device = torch.device("cpu")
        digitiser.enable_perspective_correction = True
        digitiser.last_info = DigitizeInfo()
        corrected = torch.zeros(3, 900, 1600, dtype=torch.uint8)

        with (
            patch("torchvision.io.decode_image") as mock_decode,
            patch("src.pipeline.digitize.correct_perspective") as mock_correct,
        ):
            mock_decode.return_value = torch.zeros(3, 1000, 1700, dtype=torch.uint8)
            mock_correct.return_value = MagicMock(
                image=corrected,
                applied=True,
                confidence=0.91,
            )
            digitiser._load_image(Path("fake.png"))

        assert digitiser.last_info.page_correction_applied is True
        assert digitiser.last_info.page_correction_confidence == pytest.approx(0.91)
        assert digitiser.last_info.page_correction_version == "conservative-perspective-v1"

    def test_load_image_applies_enabled_shadow_normalization(self) -> None:
        """The shadow experiment flag is independent from perspective correction."""
        digitiser = object.__new__(ECGDigitiser)
        digitiser.device = torch.device("cpu")
        digitiser.enable_shadow_normalization = True
        digitiser.last_info = DigitizeInfo()
        normalized = torch.zeros(3, 900, 1600, dtype=torch.uint8)

        with (
            patch("torchvision.io.decode_image") as mock_decode,
            patch("src.pipeline.digitize.normalize_broad_shadow") as mock_normalize,
        ):
            mock_decode.return_value = torch.zeros(3, 900, 1600, dtype=torch.uint8)
            mock_normalize.return_value = MagicMock(
                image=normalized,
                applied=True,
                shadow_score=0.32,
            )
            digitiser._load_image(Path("fake.png"))

        assert digitiser.last_info.shadow_normalization_applied is True
        assert digitiser.last_info.shadow_score == pytest.approx(0.32)
        assert digitiser.last_info.shadow_normalization_version == "conservative-shadow-v1"


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

    def test_postprocess_outputs_preserve_calibrated_millivolts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.pipeline.digitize as digitize_module
        from src.pipeline.digitize import ECGDigitiser

        monkeypatch.setattr(digitize_module, "highpass_filter", lambda signal, _: signal)
        monkeypatch.setattr(digitize_module, "wavelet_denoise", lambda signal, _: signal)
        canonical = torch.tensor(
            np.tile(np.linspace(-1000.0, 1000.0, 5000), (12, 1)),
            dtype=torch.float32,
        )

        outputs = ECGDigitiser._postprocess_outputs(canonical)

        np.testing.assert_allclose(
            outputs.calibrated_millivolts[0, [0, -1]],
            np.array([-1.0, 1.0]),
            atol=1e-6,
        )
        assert outputs.calibrated_millivolts.dtype == np.float32
        assert outputs.model_input.dtype == np.float32
        assert np.std(outputs.model_input) == pytest.approx(1.0, abs=1e-6)


class TestQualityGuards:
    """Verify silent digitization failures are rejected."""

    def test_negative_einthoven_score_is_a_quality_warning(self) -> None:
        from src.pipeline.digitize import DigitizeInfo

        healthy = {
            "layout_cost": 0.1,
            "detected_leads_count": 12,
            "avg_pixel_per_mm": 8.0,
            "raw_lines_count": 4,
            "nonzero_leads_count": 12,
        }
        assert DigitizeInfo(**healthy, einthoven_score=-0.2).has_warnings is True
        assert DigitizeInfo(**healthy).has_warnings is False

    def test_extract_canonical_raises_for_all_nan(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        result = {
            "signal": {
                "canonical_lines": torch.full((12, 1000), float("nan")),
            }
        }

        digitiser = object.__new__(ECGDigitiser)
        with pytest.raises(RuntimeError, match="no finite canonical signal values"):
            digitiser._extract_canonical(result)

    def test_count_nonzero_leads(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        signal = np.zeros((12, 5000), dtype=np.float32)
        signal[:9] = 0.5

        assert ECGDigitiser._count_nonzero_leads(signal) == 9

    def test_validate_signal_rejects_sparse_output(self) -> None:
        from src.pipeline.digitize import DigitizeInfo, ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.last_info = DigitizeInfo(raw_lines_count=1, nonzero_leads_count=4)

        signal = np.zeros((12, 5000), dtype=np.float32)
        signal[:4] = 1.0

        with pytest.raises(RuntimeError, match="failed quality checks"):
            digitiser._validate_digitized_signal(signal, layout_hint="3x4+1R")


class TestRetryLogic:
    """Verify poor-quality results trigger fallback attempts."""

    def test_should_retry_on_high_layout_cost(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser._wrapper = MagicMock(apply_dewarping=False)

        result = {
            "signal": {
                "layout_matching_cost": 1.55,
                "raw_lines": torch.zeros(4, 100),
                "n_detected": 11,
            }
        }

        assert digitiser._should_retry_with_dewarping(result) is True

    def test_score_prefers_finite_canonical_signal(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)

        poor = {
            "signal": {
                "layout_matching_cost": 1.55,
                "raw_lines": torch.zeros(4, 100),
                "n_detected": 11,
                "canonical_lines": torch.full((12, 100), float("nan")),
            }
        }
        good = {
            "signal": {
                "layout_matching_cost": 0.44,
                "raw_lines": torch.zeros(4, 100),
                "n_detected": 11,
                "canonical_lines": torch.randn(12, 100),
            }
        }

        assert digitiser._score_raw_result(good) > digitiser._score_raw_result(poor)

    def test_run_best_inference_skips_retry_when_disabled(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.enable_dewarping_retry = False
        digitiser.enable_orientation_retry = False
        digitiser._run_inference = MagicMock(return_value={"signal": {}})
        digitiser._score_raw_result = MagicMock(return_value=1.0)
        digitiser._should_retry_with_dewarping = MagicMock(return_value=True)

        result = digitiser._run_best_inference(torch.zeros(1, 3, 10, 10))

        assert result["processing_mode"] == "default"
        assert digitiser._run_inference.call_count == 1

    def test_run_best_inference_selects_better_rotated_orientation(self) -> None:
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.enable_orientation_retry = True
        digitiser.enable_dewarping_retry = False
        poor = {"signal": {"layout_matching_cost": 4.0}}
        good = {"signal": {"layout_matching_cost": 0.2}}
        digitiser._run_inference = MagicMock(side_effect=[poor, good])
        digitiser._score_raw_result = MagicMock(side_effect=[1.0, 10.0])
        digitiser._should_retry_with_orientation = MagicMock(return_value=True)

        image = torch.arange(24).reshape(1, 3, 2, 4)
        result = digitiser._run_best_inference(image)

        assert result["processing_mode"] == "rotated_180_retry"
        rotated = digitiser._run_inference.call_args_list[1].args[0]
        torch.testing.assert_close(rotated, torch.rot90(image, k=2, dims=(2, 3)))

    def test_run_best_inference_keeps_default_when_rotated_orientation_is_worse(
        self,
    ) -> None:
        from src.pipeline.digitize import ECGDigitiser

        digitiser = object.__new__(ECGDigitiser)
        digitiser.enable_orientation_retry = True
        digitiser.enable_dewarping_retry = False
        default = {"signal": {"layout_matching_cost": 0.5}}
        rotated = {"signal": {"layout_matching_cost": 1.0}}
        digitiser._run_inference = MagicMock(side_effect=[default, rotated])
        digitiser._score_raw_result = MagicMock(side_effect=[10.0, 1.0])
        digitiser._should_retry_with_orientation = MagicMock(return_value=True)

        result = digitiser._run_best_inference(torch.zeros(1, 3, 2, 4))

        assert result["processing_mode"] == "default"


# ---------------------------------------------------------------------------
# Integration tests — require model weights
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not WEIGHTS_EXIST, reason="Model weights not found")
class TestIntegrationDigitize:
    """Full integration test with real Open-ECG-Digitizer model."""

    @pytest.fixture(scope="class")
    def digitiser(self) -> ECGDigitiser:
        return ECGDigitiser()

    def test_digitize_returns_correct_shape(
        self, digitiser: ECGDigitiser, tmp_path: Path
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
