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
