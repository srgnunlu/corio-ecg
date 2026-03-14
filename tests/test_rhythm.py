# Tests for heart-rate estimation and diagnosis post-processing.

import numpy as np

from src.pipeline.diagnose import ECGDiagnoser
from src.utils.ecg_labels import ECG_FOUNDER_LABELS
from src.utils.rhythm import estimate_heart_rate_bpm


def _make_pulse_signal(bpm: float, sample_rate: int = 500) -> np.ndarray:
    """Create a simple 10-second multi-lead pulse train."""
    n_samples = sample_rate * 10
    signal = np.zeros((12, n_samples), dtype=np.float32)
    rr = int(round((60.0 / bpm) * sample_rate))
    pulse = np.array([0.4, 1.2, 2.2, 1.2, 0.4], dtype=np.float32)

    for peak in range(rr, n_samples - len(pulse), rr):
        signal[1, peak : peak + len(pulse)] = pulse
        signal[6, peak : peak + len(pulse)] = pulse * 0.7

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return (signal - mean) / (std + 1e-8)


class TestHeartRateEstimation:
    def test_estimates_tachycardia_rate(self) -> None:
        signal = _make_pulse_signal(150.0)
        hr = estimate_heart_rate_bpm(signal)
        assert hr is not None
        assert 140.0 <= hr <= 160.0

    def test_estimates_bradycardia_rate(self) -> None:
        signal = _make_pulse_signal(42.0)
        hr = estimate_heart_rate_bpm(signal)
        assert hr is not None
        assert 38.0 <= hr <= 46.0


class TestDiagnosisRateConsistency:
    def test_downweights_brady_labels_on_fast_signal(self) -> None:
        signal = _make_pulse_signal(160.0)
        probabilities = np.ones(len(ECG_FOUNDER_LABELS), dtype=np.float32)

        diagnoser = object.__new__(ECGDiagnoser)
        diagnoser.last_estimated_hr_bpm = None
        adjusted = diagnoser._apply_rate_consistency_adjustments(probabilities, signal)

        brady_idx = ECG_FOUNDER_LABELS.index("SINUS BRADYCARDIA")
        tachy_idx = ECG_FOUNDER_LABELS.index("SINUS TACHYCARDIA")
        assert adjusted[brady_idx] < 0.1
        assert adjusted[tachy_idx] > adjusted[brady_idx]
