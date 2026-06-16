# Tests for heart-rate estimation and diagnosis post-processing.

import numpy as np

from src.pipeline.diagnose import ECGDiagnoser
from src.pipeline.digitize import _extract_rhythm_strip, _process_rhythm_strip
from src.utils.ecg_labels import ECG_FOUNDER_LABELS
from src.utils.rhythm import estimate_heart_rate_bpm, estimate_rhythm_hr


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


def _make_rhythm_strip(
    bpm: float,
    *,
    jitter: float = 0.0,
    seed: int = 0,
    lead_idx: int = 1,
    sample_rate: int = 500,
) -> np.ndarray:
    """Build a z-scored (12, 5000) strip with QRS pulses on a single lead.

    With jitter > 0 the RR intervals vary beat-to-beat, mimicking an irregular
    rhythm (atrial fibrillation) where only the full strip carries the true rate.
    """
    n_samples = sample_rate * 10
    signal = np.zeros((12, n_samples), dtype=np.float64)
    rng = np.random.default_rng(seed)
    base_rr = int(round((60.0 / bpm) * sample_rate))
    samples = np.arange(n_samples)

    lead = np.zeros(n_samples)
    position = base_rr
    while position < n_samples - 10:
        lead += 2.2 * np.exp(-0.5 * ((samples - position) / 4.0) ** 2)
        step = base_rr + int(rng.normal(0.0, jitter * base_rr))
        position += max(step, int(sample_rate * 0.3))
    signal[lead_idx] = lead

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return ((signal - mean) / (std + 1e-8)).astype(np.float32)


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


class TestRhythmStripHeartRate:
    def test_estimates_bradycardia(self) -> None:
        hr = estimate_rhythm_hr(_make_rhythm_strip(45.0))
        assert hr is not None
        assert 42.0 <= hr <= 48.0

    def test_estimates_normal_rate(self) -> None:
        hr = estimate_rhythm_hr(_make_rhythm_strip(72.0))
        assert hr is not None
        assert 68.0 <= hr <= 76.0

    def test_estimates_tachycardia(self) -> None:
        hr = estimate_rhythm_hr(_make_rhythm_strip(150.0))
        assert hr is not None
        assert 142.0 <= hr <= 158.0

    def test_irregular_rhythm_returns_ventricular_rate(self) -> None:
        # An AF-like strip (irregular RR) should still yield a plausible mean
        # ventricular rate from the median RR over the full 10 s window.
        hr = estimate_rhythm_hr(_make_rhythm_strip(90.0, jitter=0.25, seed=3))
        assert hr is not None
        assert 70.0 <= hr <= 115.0

    def test_prefers_lead_two(self) -> None:
        # Lead II carries the true rate; a decoy on another lead has a different
        # rate. Estimation must lock onto Lead II (index 1).
        strip = _make_rhythm_strip(60.0, lead_idx=1)
        decoy = _make_rhythm_strip(120.0, lead_idx=7)
        strip[7] = decoy[7]
        hr = estimate_rhythm_hr(strip)
        assert hr is not None
        assert 56.0 <= hr <= 64.0

    def test_returns_none_on_flat_strip(self) -> None:
        assert estimate_rhythm_hr(np.zeros((12, 5000), dtype=np.float32)) is None


class TestRhythmStripExtraction:
    def test_preserves_full_width_lead_and_zeroes_short_segments(self) -> None:
        total_len = 10000
        signal = np.full((12, total_len), np.nan, dtype=np.float64)
        # Lead II spans the full page width (rhythm strip).
        signal[1] = np.sin(np.linspace(0.0, 80.0 * np.pi, total_len))
        # Lead I and V1 are short ~2.5 s printed segments.
        signal[0, 0:2500] = 1.0
        signal[6, 2500:5000] = 1.0

        strip = _extract_rhythm_strip(signal)
        assert strip is not None
        assert np.isfinite(strip[1]).all()
        assert strip[1].std() > 0.0
        assert strip[0].std() == 0.0
        assert strip[6].std() == 0.0

    def test_returns_none_without_full_width_lead(self) -> None:
        signal = np.full((12, 10000), np.nan, dtype=np.float64)
        for lead_idx in range(12):
            start = (lead_idx % 4) * 2500
            signal[lead_idx, start : start + 2500] = 1.0
        assert _extract_rhythm_strip(signal) is None

    def test_processed_strip_is_model_ready_and_hr_recoverable(self) -> None:
        total_len = 10000
        signal = np.full((12, total_len), np.nan, dtype=np.float64)
        # 60 bpm QRS train on Lead II at the canonical microvolt scale.
        samples = np.arange(total_len)
        rr = int(total_len / 10.0 * (60.0 / 60.0))  # 1 s at 1000 Hz canonical grid
        lead = np.zeros(total_len)
        for peak in range(rr, total_len - 10, rr):
            lead += 1500.0 * np.exp(-0.5 * ((samples - peak) / 8.0) ** 2)
        signal[1] = lead

        strip = _extract_rhythm_strip(signal)
        assert strip is not None
        processed = _process_rhythm_strip(strip)
        assert processed.shape == (12, 5000)
        hr = estimate_rhythm_hr(processed)
        assert hr is not None
        assert 55.0 <= hr <= 65.0


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

    def test_rhythm_strip_overrides_tiled_signal_for_hr(self) -> None:
        # Tiled signal carries no usable rate; the rhythm strip says tachycardia.
        # The adjustment must follow the strip and suppress bradycardia labels.
        flat_signal = np.zeros((12, 5000), dtype=np.float32)
        rhythm_strip = _make_rhythm_strip(150.0)
        probabilities = np.ones(len(ECG_FOUNDER_LABELS), dtype=np.float32)

        diagnoser = object.__new__(ECGDiagnoser)
        diagnoser.last_estimated_hr_bpm = None
        adjusted = diagnoser._apply_rate_consistency_adjustments(
            probabilities, flat_signal, rhythm_strip=rhythm_strip
        )

        assert diagnoser.last_estimated_hr_bpm is not None
        assert 142.0 <= diagnoser.last_estimated_hr_bpm <= 158.0
        brady_idx = ECG_FOUNDER_LABELS.index("SINUS BRADYCARDIA")
        assert adjusted[brady_idx] < 0.1
