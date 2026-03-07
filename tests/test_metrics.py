# Tests for signal quality metrics (SNR and Pearson correlation)

import numpy as np
import pytest

from src.utils.metrics import compute_pearson_per_lead, compute_snr

# Fixed seed for reproducible random tests
RNG = np.random.default_rng(42)

# Shared test signal: 12 leads, 5000 samples
CLEAN_SIGNAL = RNG.standard_normal((12, 5000))


class TestComputeSNR:
    """Tests for compute_snr function."""

    def test_snr_identical_signals(self) -> None:
        """Identical signals should produce very high (infinite) SNR."""
        snr = compute_snr(CLEAN_SIGNAL, CLEAN_SIGNAL.copy())
        assert snr > 50, f"Expected SNR > 50 dB for identical signals, got {snr}"

    def test_snr_noisy_signal(self) -> None:
        """Adding moderate noise should give SNR in a predictable range."""
        # Add noise at 10% of signal standard deviation (~20 dB expected)
        noise_level = 0.1
        noise = RNG.standard_normal(CLEAN_SIGNAL.shape) * noise_level
        digitized = CLEAN_SIGNAL + noise
        snr = compute_snr(CLEAN_SIGNAL, digitized)
        # 10*log10(1/0.01) = 20 dB, allow some tolerance
        assert 15 < snr < 25, f"Expected SNR ~20 dB, got {snr}"

    def test_snr_very_noisy(self) -> None:
        """Heavy noise should produce low SNR (<5 dB)."""
        # Noise power equal to signal power (~0 dB)
        noise = RNG.standard_normal(CLEAN_SIGNAL.shape) * 1.0
        digitized = CLEAN_SIGNAL + noise
        snr = compute_snr(CLEAN_SIGNAL, digitized)
        assert snr < 5, f"Expected SNR < 5 dB for heavy noise, got {snr}"


class TestComputePearsonPerLead:
    """Tests for compute_pearson_per_lead function."""

    def test_pearson_identical_signals(self) -> None:
        """Identical signals should have correlation ~1.0 for all leads."""
        correlations = compute_pearson_per_lead(CLEAN_SIGNAL, CLEAN_SIGNAL.copy())
        for i, r in enumerate(correlations):
            assert r == pytest.approx(1.0, abs=1e-10), (
                f"Lead {i}: expected r=1.0, got {r}"
            )

    def test_pearson_uncorrelated(self) -> None:
        """Independent random signals should have correlations near 0."""
        random_signal = RNG.standard_normal(CLEAN_SIGNAL.shape)
        correlations = compute_pearson_per_lead(CLEAN_SIGNAL, random_signal)
        for i, r in enumerate(correlations):
            assert abs(r) < 0.1, (
                f"Lead {i}: expected |r| < 0.1 for uncorrelated signals, got {r}"
            )

    def test_pearson_returns_12_values(self) -> None:
        """Should return exactly 12 correlation values."""
        correlations = compute_pearson_per_lead(CLEAN_SIGNAL, CLEAN_SIGNAL.copy())
        assert len(correlations) == 12, (
            f"Expected 12 values, got {len(correlations)}"
        )

    def test_pearson_constant_lead_returns_zero(self) -> None:
        """Constant (zero-padded) leads should return 0.0 instead of NaN."""
        signal_with_constant = CLEAN_SIGNAL.copy()
        signal_with_constant[0] = 0.0  # zero out first lead
        correlations = compute_pearson_per_lead(signal_with_constant, CLEAN_SIGNAL)
        assert correlations[0] == 0.0, (
            f"Expected 0.0 for constant lead, got {correlations[0]}"
        )
        assert not np.isnan(correlations[0]), "Should not be NaN"
