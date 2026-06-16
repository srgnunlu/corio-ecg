# Tests for ECG signal cleaning utilities
# Verifies high-pass filter, wavelet denoising, and Einthoven consistency

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.signal_clean import (
    einthoven_consistency,
    goldberger_consistency,
    highpass_filter,
    wavelet_denoise,
)

SAMPLE_RATE = 500
N_SAMPLES = 5000


class TestHighpassFilter:
    """Verify baseline wander removal preserves diagnostic content."""

    def test_removes_baseline_wander(self) -> None:
        """0.1 Hz baseline drift should be attenuated significantly."""
        t = np.linspace(0, 10, N_SAMPLES)
        baseline = 2.0 * np.sin(2 * np.pi * 0.1 * t)
        qrs = np.sin(2 * np.pi * 10 * t)
        signal = np.stack([baseline + qrs] * 12)

        filtered = highpass_filter(signal, SAMPLE_RATE)

        # Baseline energy should be reduced significantly
        baseline_residual = np.mean(np.abs(filtered - np.stack([qrs] * 12)))
        baseline_original = np.mean(np.abs(np.stack([baseline] * 12)))
        assert baseline_residual < baseline_original * 0.3

    def test_preserves_qrs_frequency_content(self) -> None:
        """10 Hz signal (QRS range) should pass through with >95% correlation."""
        t = np.linspace(0, 10, N_SAMPLES)
        qrs = np.sin(2 * np.pi * 10 * t)
        signal = np.stack([qrs] * 12)

        filtered = highpass_filter(signal, SAMPLE_RATE)

        for i in range(12):
            corr = np.corrcoef(filtered[i], qrs)[0, 1]
            assert corr > 0.95, f"Lead {i}: correlation {corr:.3f} too low"

    def test_flat_lead_unchanged(self) -> None:
        """Flat (zero) leads should pass through without errors."""
        signal = np.zeros((12, N_SAMPLES))
        filtered = highpass_filter(signal, SAMPLE_RATE)
        np.testing.assert_array_equal(filtered, signal)


class TestWaveletDenoise:
    """Verify wavelet denoising removes noise while preserving morphology."""

    def test_reduces_gaussian_noise(self) -> None:
        """Random noise should be reduced by at least 40%."""
        rng = np.random.RandomState(42)
        t = np.linspace(0, 10, N_SAMPLES)
        clean = np.sin(2 * np.pi * 1.2 * t)
        noise = 0.3 * rng.randn(N_SAMPLES)
        noisy = np.stack([clean + noise] * 12)
        clean_12 = np.stack([clean] * 12)

        denoised = wavelet_denoise(noisy, SAMPLE_RATE)

        mse_before = np.mean((noisy - clean_12) ** 2)
        mse_after = np.mean((denoised - clean_12) ** 2)
        assert mse_after < mse_before * 0.6, (
            f"Noise reduction insufficient: {mse_before:.4f} -> {mse_after:.4f}"
        )

    def test_preserves_peak_amplitude(self) -> None:
        """QRS-like peak amplitudes should not be reduced by more than 20%."""
        rng = np.random.RandomState(42)
        clean = np.zeros(N_SAMPLES)
        # Simulate QRS complexes: ~40ms wide Gaussian peaks (~72 bpm)
        for peak_center in range(350, N_SAMPLES, 700):
            for offset in range(-10, 11):
                idx = peak_center + offset
                if 0 <= idx < N_SAMPLES:
                    clean[idx] = 3.0 * np.exp(-0.5 * (offset / 4.0) ** 2)
        noise = 0.1 * rng.randn(N_SAMPLES)
        noisy = np.stack([clean + noise] * 12)

        denoised = wavelet_denoise(noisy, SAMPLE_RATE)

        original_max = np.max(np.abs(noisy[0]))
        denoised_max = np.max(np.abs(denoised[0]))
        ratio = denoised_max / original_max
        assert ratio > 0.8, f"Peak amplitude lost: {ratio:.2f}"

    def test_flat_lead_unchanged(self) -> None:
        """Flat (zero) leads should pass through without errors."""
        signal = np.zeros((12, N_SAMPLES))
        denoised = wavelet_denoise(signal, SAMPLE_RATE)
        np.testing.assert_array_equal(denoised, signal)

    def test_output_shape_matches_input(self) -> None:
        """Output shape must equal input shape exactly."""
        rng = np.random.RandomState(42)
        signal = rng.randn(12, N_SAMPLES)
        denoised = wavelet_denoise(signal, SAMPLE_RATE)
        assert denoised.shape == signal.shape

    def test_short_signal_handled(self) -> None:
        """Very short signals should not crash wavelet decomposition."""
        rng = np.random.RandomState(42)
        short = rng.randn(12, 200)
        denoised = wavelet_denoise(short, SAMPLE_RATE)
        assert denoised.shape == short.shape


class TestEinthovenConsistency:
    """Verify Einthoven's law detection for lead assignment quality."""

    def test_perfect_consistency(self) -> None:
        """When II = I + III exactly, score should be ~1.0."""
        rng = np.random.RandomState(42)
        signal = np.zeros((12, N_SAMPLES))
        signal[0] = rng.randn(N_SAMPLES)
        signal[2] = rng.randn(N_SAMPLES)
        signal[1] = signal[0] + signal[2]

        score = einthoven_consistency(signal)
        assert score > 0.99, f"Perfect consistency score too low: {score:.3f}"

    def test_detects_misassigned_leads(self) -> None:
        """Random leads should give low consistency score."""
        rng = np.random.RandomState(42)
        signal = rng.randn(12, N_SAMPLES)

        score = einthoven_consistency(signal)
        assert score < 0.5, f"Random leads scored too high: {score:.3f}"

    def test_flat_leads_return_zero(self) -> None:
        """Flat leads should return 0.0, not NaN or crash."""
        signal = np.zeros((12, N_SAMPLES))
        score = einthoven_consistency(signal)
        assert score == 0.0

    def test_works_after_z_score(self) -> None:
        """Einthoven check should work on z-score normalized signals."""
        rng = np.random.RandomState(42)
        signal = np.zeros((12, N_SAMPLES))
        signal[0] = rng.randn(N_SAMPLES) * 2.0
        signal[2] = rng.randn(N_SAMPLES) * 1.5
        signal[1] = signal[0] + signal[2]
        for i in range(3, 12):
            signal[i] = rng.randn(N_SAMPLES)

        mean = np.mean(signal)
        std = np.std(signal)
        normalized = (signal - mean) / (std + 1e-8)

        score = einthoven_consistency(normalized)
        assert score > 0.95, f"Post-z-score consistency too low: {score:.3f}"


def _make_consistent_limb_leads(rng: np.random.RandomState) -> np.ndarray:
    """Build a 12-lead signal whose limb leads satisfy every derivation rule."""
    signal = np.zeros((12, N_SAMPLES))
    signal[0] = rng.randn(N_SAMPLES)              # I
    signal[2] = rng.randn(N_SAMPLES)              # III
    signal[1] = signal[0] + signal[2]             # II  = I + III
    signal[3] = -(signal[0] + signal[1]) / 2.0    # aVR = -(I + II) / 2
    signal[4] = (signal[0] - signal[2]) / 2.0     # aVL = (I - III) / 2
    signal[5] = (signal[1] + signal[2]) / 2.0     # aVF = (II + III) / 2
    for i in range(6, 12):
        signal[i] = rng.randn(N_SAMPLES)          # chest leads: independent
    return signal


class TestGoldbergerConsistency:
    """Verify the full Einthoven/Goldberger limb-lead redundancy check."""

    def test_perfect_consistency(self) -> None:
        """Exact derivation identities give ~1.0 correlations and ~0 residuals."""
        rng = np.random.RandomState(42)
        signal = _make_consistent_limb_leads(rng)

        result = goldberger_consistency(signal)

        for rule, corr in result.correlations.items():
            assert corr > 0.99, f"Rule {rule} correlation too low: {corr:.3f}"
        assert result.mean_residual < 0.01
        assert result.worst_residual < 0.01

    def test_recovers_phase_shifted_lead(self) -> None:
        """A circularly shifted aVR still matches under the best-lag search.

        On a 3x4 paper ECG the augmented leads come from a different column
        (time window) than I/II/III, so the measured trace is phase-shifted
        from its derivation. The lag-tolerant metric must see through that.
        """
        rng = np.random.RandomState(7)
        signal = _make_consistent_limb_leads(rng)
        shift = 300  # 0.6 s at 500 Hz — within one slow RR interval
        signal[3] = np.roll(signal[3], shift)

        lag_tolerant = goldberger_consistency(signal, max_lag_seconds=1.3)
        zero_lag = goldberger_consistency(signal, max_lag_seconds=0.0)

        # Circularly rolling white noise loses ``shift`` samples of overlap in
        # the linear cross-correlation, capping recovery below 1.0; the contrast
        # with the zero-lag value is what proves phase tolerance works.
        assert lag_tolerant.correlations["aVR"] > 0.9
        assert zero_lag.correlations["aVR"] < 0.5
        assert lag_tolerant.residuals["aVR"] < zero_lag.residuals["aVR"]

    def test_detects_inconsistent_leads(self) -> None:
        """Fully random limb leads give high residuals."""
        rng = np.random.RandomState(1)
        signal = rng.randn(12, N_SAMPLES)

        result = goldberger_consistency(signal)

        assert result.mean_residual > 0.5
        assert result.worst_residual >= result.mean_residual

    def test_flat_leads_return_max_residual(self) -> None:
        """Flat leads yield 0 correlation and residual 1.0 without crashing."""
        signal = np.zeros((12, N_SAMPLES))
        result = goldberger_consistency(signal)
        assert result.mean_residual == 1.0
        assert all(corr == 0.0 for corr in result.correlations.values())

    def test_requires_six_limb_leads(self) -> None:
        """Fewer than six leads cannot define the augmented derivations."""
        with pytest.raises(ValueError, match="needs >=6 leads"):
            goldberger_consistency(np.zeros((3, N_SAMPLES)))

    def test_worst_residual_is_max_of_rules(self) -> None:
        """The worst residual must equal the maximum per-rule residual."""
        rng = np.random.RandomState(99)
        signal = _make_consistent_limb_leads(rng)
        signal[4] = rng.randn(N_SAMPLES)  # corrupt only aVL

        result = goldberger_consistency(signal)

        assert result.worst_residual == max(result.residuals.values())
        assert result.residuals["aVL"] == result.worst_residual
