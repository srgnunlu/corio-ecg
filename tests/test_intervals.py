# Tests for PR/QRS/QT/QTc interval measurement on synthetic known-interval ECGs.

import numpy as np

from src.measurement.delineation import detect_rpeaks
from src.measurement.intervals import (
    IntervalMeasurements,
    interpret_intervals,
    measure_intervals,
)

SAMPLE_RATE = 500


def _triangle(lead: np.ndarray, onset: int, peak: int, offset: int, amp: float) -> None:
    """Add a triangular wave (linear up to peak, linear down to offset) in place.

    Triangles give piecewise-linear limbs, so threshold/tangent delineation
    recovers the geometric onset/offset cleanly — exact ground truth for tests.
    """
    n = lead.size
    if peak > onset:
        for i in range(max(onset, 0), min(peak, n)):
            lead[i] += amp * (i - onset) / (peak - onset)
    if offset > peak:
        for i in range(max(peak, 0), min(offset, n)):
            lead[i] += amp * (1.0 - (i - peak) / (offset - peak))


def _synthetic_ecg(
    *,
    bpm: float = 60.0,
    pr_ms: float = 160.0,
    qrs_ms: float = 90.0,
    qt_ms: float = 380.0,
    pr_tail_amp: float = 0.0,
    jitter: float = 0.0,
    seed: int = 0,
    lead_idx: int = 1,
    n_samples: int = SAMPLE_RATE * 10,
) -> np.ndarray:
    """Build a z-scored (12, N) ECG with explicit fiducials on one lead.

    Each beat is anchored at QRS onset: P spans [qrs_on - PR, ...], QRS spans
    [qrs_on, qrs_on + QRS] with R at its center, and T ends at qrs_on + QT. The
    geometric onsets/ends equal the target intervals, so measurements can be
    checked against truth.
    """
    rng = np.random.default_rng(seed)
    lead = np.zeros(n_samples, dtype=np.float64)
    base_rr = (60.0 / bpm) * SAMPLE_RATE

    def ms(value: float) -> int:
        return int(round(value / 1000.0 * SAMPLE_RATE))

    qrs_n, pr_n, qt_n = ms(qrs_ms), ms(pr_ms), ms(qt_ms)
    p_dur = ms(100.0)

    position = base_rr
    while position < n_samples - base_rr * 0.5:
        qrs_on = int(round(position))
        # P wave: ends ~30 ms before QRS onset, onset = qrs_on - PR.
        p_on = qrs_on - pr_n
        p_off = p_on + p_dur
        if pr_tail_amp > 0.0:
            tail_on = qrs_on - ms(250.0)
            lead[max(tail_on, 0):max(p_on, 0)] += pr_tail_amp
        _triangle(lead, p_on, (p_on + p_off) // 2, p_off, amp=0.18)
        # QRS: sharp triangle, R peak at the center.
        _triangle(lead, qrs_on, qrs_on + qrs_n // 2, qrs_on + qrs_n, amp=1.6)
        # T wave: starts after the ST segment, ends exactly at qrs_on + QT.
        t_end = qrs_on + qt_n
        t_on = qrs_on + qrs_n + ms(80.0)
        _triangle(lead, t_on, (t_on + t_end) // 2, t_end, amp=0.40)
        position += max(base_rr + rng.normal(0.0, jitter * base_rr), SAMPLE_RATE * 0.3)

    signal = np.zeros((12, n_samples), dtype=np.float64)
    signal[lead_idx] = lead
    mean, std = float(np.mean(signal)), float(np.std(signal))
    return ((signal - mean) / (std + 1e-8)).astype(np.float32)


class TestRPeakDetection:
    def test_counts_beats_at_60bpm(self) -> None:
        ecg = _synthetic_ecg(bpm=60.0)
        peaks = detect_rpeaks(ecg[1], SAMPLE_RATE)
        # ~10 beats in 10 s; allow edge effects.
        assert 8 <= peaks.size <= 11

    def test_returns_empty_on_flat_lead(self) -> None:
        flat = np.zeros(SAMPLE_RATE * 10, dtype=np.float64)
        assert detect_rpeaks(flat, SAMPLE_RATE).size == 0


class TestIntervalMeasurement:
    def test_recovers_normal_intervals(self) -> None:
        ecg = _synthetic_ecg(bpm=60.0, pr_ms=160.0, qrs_ms=90.0, qt_ms=380.0)
        m = measure_intervals(ecg)

        assert m.heart_rate_bpm is not None
        assert abs(m.heart_rate_bpm - 60.0) < 5.0
        assert m.pr_ms is not None and abs(m.pr_ms - 160.0) < 30.0
        assert m.qrs_ms is not None and abs(m.qrs_ms - 90.0) < 30.0
        assert m.qt_ms is not None and abs(m.qt_ms - 380.0) < 40.0
        assert m.measured_lead == "II"

    def test_pr_onset_does_not_stick_to_search_boundary(self) -> None:
        # Digitized paper traces often carry residual T-tail/baseline energy before
        # the P wave. PR should follow the P wave itself, not the left edge of the
        # 250 ms search window.
        ecg = _synthetic_ecg(bpm=60.0, pr_ms=150.0, qrs_ms=80.0, pr_tail_amp=0.10)
        m = measure_intervals(ecg)

        assert m.pr_ms is not None
        assert abs(m.pr_ms - 150.0) < 35.0

    def test_qtc_bazett_equals_qt_at_60bpm(self) -> None:
        # At 60 bpm RR = 1 s, so Bazett QTc == QT.
        ecg = _synthetic_ecg(bpm=60.0, qt_ms=400.0)
        m = measure_intervals(ecg)
        assert m.qtc_bazett_ms is not None
        assert abs(m.qtc_bazett_ms - m.qt_ms) < 1.0
        assert m.qtc_formula == "bazett"

    def test_prefers_fridericia_at_tachycardia(self) -> None:
        ecg = _synthetic_ecg(bpm=120.0, qt_ms=320.0)
        m = measure_intervals(ecg)
        assert m.qtc_formula == "fridericia"
        assert m.qtc_preferred_ms == m.qtc_fridericia_ms

    def test_uses_rhythm_strip_when_provided(self) -> None:
        signal = _synthetic_ecg(bpm=75.0, qt_ms=360.0)
        strip = _synthetic_ecg(bpm=50.0, qt_ms=400.0)
        m = measure_intervals(signal, rhythm_strip=strip)
        # RR must come from the strip (50 bpm), not the tiled signal (75 bpm).
        assert m.heart_rate_bpm is not None
        assert abs(m.heart_rate_bpm - 50.0) < 6.0

    def test_unmeasurable_on_flat_signal(self) -> None:
        flat = np.zeros((12, SAMPLE_RATE * 10), dtype=np.float32)
        m = measure_intervals(flat)
        assert m.quality == "unmeasurable"
        assert m.qt_ms is None
        assert m.heart_rate_bpm is None


class TestInterpretation:
    def test_flags_prolonged_pr(self) -> None:
        m = IntervalMeasurements(
            heart_rate_bpm=60.0, rr_ms=1000.0, pr_ms=240.0, qrs_ms=90.0,
            qt_ms=400.0, qtc_bazett_ms=400.0, qtc_fridericia_ms=400.0,
            qtc_preferred_ms=400.0, qtc_formula="bazett", measured_lead="II",
            n_beats=10, quality="good",
        )
        flags = interpret_intervals(m)
        assert "first-degree AV block" in flags["pr"]
        assert flags["qrs"] == "normal"

    def test_flags_wide_qrs_and_prolonged_qtc(self) -> None:
        m = IntervalMeasurements(
            heart_rate_bpm=60.0, rr_ms=1000.0, pr_ms=160.0, qrs_ms=140.0,
            qt_ms=480.0, qtc_bazett_ms=480.0, qtc_fridericia_ms=480.0,
            qtc_preferred_ms=480.0, qtc_formula="bazett", measured_lead="II",
            n_beats=10, quality="good",
        )
        flags = interpret_intervals(m, sex="male")
        assert "bundle branch block" in flags["qrs"]
        assert "prolonged" in flags["qtc"]

    def test_sex_specific_qtc_cutoff(self) -> None:
        m = IntervalMeasurements(
            heart_rate_bpm=60.0, rr_ms=1000.0, pr_ms=160.0, qrs_ms=90.0,
            qt_ms=455.0, qtc_bazett_ms=455.0, qtc_fridericia_ms=455.0,
            qtc_preferred_ms=455.0, qtc_formula="bazett", measured_lead="II",
            n_beats=10, quality="good",
        )
        # 455 ms: prolonged for males (>450), normal for females (>460).
        assert "prolonged" in interpret_intervals(m, sex="male")["qtc"]
        assert interpret_intervals(m, sex="female")["qtc"] == "normal"
