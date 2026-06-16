# Tests for rule-based rhythm classification (sinus / AF / rate / PVC).

import numpy as np

from src.measurement.rhythm_analysis import analyze_rhythm

SAMPLE_RATE = 500


def _triangle(lead: np.ndarray, onset: int, peak: int, offset: int, amp: float) -> None:
    n = lead.size
    if peak > onset:
        for i in range(max(onset, 0), min(peak, n)):
            lead[i] += amp * (i - onset) / (peak - onset)
    if offset > peak:
        for i in range(max(peak, 0), min(offset, n)):
            lead[i] += amp * (1.0 - (i - peak) / (offset - peak))


def _ms(value: float) -> int:
    return int(round(value / 1000.0 * SAMPLE_RATE))


def _add_beat(
    lead: np.ndarray,
    qrs_on: int,
    *,
    qrs_ms: float = 90.0,
    qrs_amp: float = 1.6,
    pr_ms: float = 160.0,
    qt_ms: float = 380.0,
    include_p: bool = True,
) -> None:
    """Place one beat anchored at QRS onset (P before, QRS, then T)."""
    qrs_n, qt_n = _ms(qrs_ms), _ms(qt_ms)
    if include_p:
        p_dur = _ms(100.0)
        p_on = qrs_on - _ms(pr_ms)
        _triangle(lead, p_on, p_on + p_dur // 2, p_on + p_dur, amp=0.18)
    _triangle(lead, qrs_on, qrs_on + qrs_n // 2, qrs_on + qrs_n, amp=qrs_amp)
    t_on = qrs_on + qrs_n + _ms(80.0)
    t_end = qrs_on + qt_n
    _triangle(lead, t_on, (t_on + t_end) // 2, t_end, amp=0.40)


def _zscore_12lead(lead: np.ndarray, lead_idx: int = 1) -> np.ndarray:
    signal = np.zeros((12, lead.size), dtype=np.float64)
    signal[lead_idx] = lead
    mean, std = float(np.mean(signal)), float(np.std(signal))
    return ((signal - mean) / (std + 1e-8)).astype(np.float32)


def _regular_strip(
    bpm: float = 72.0,
    *,
    include_p: bool = True,
    n_samples: int = SAMPLE_RATE * 10,
) -> np.ndarray:
    lead = np.zeros(n_samples, dtype=np.float64)
    rr = (60.0 / bpm) * SAMPLE_RATE
    pos = rr
    while pos < n_samples - rr * 0.5:
        _add_beat(lead, int(round(pos)), include_p=include_p)
        pos += rr
    return _zscore_12lead(lead)


def _irregular_strip(
    mean_bpm: float = 90.0,
    *,
    include_p: bool,
    cv: float = 0.25,
    seed: int = 3,
    n_samples: int = SAMPLE_RATE * 10,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lead = np.zeros(n_samples, dtype=np.float64)
    base_rr = (60.0 / mean_bpm) * SAMPLE_RATE
    pos = base_rr
    while pos < n_samples - base_rr * 0.6:
        _add_beat(lead, int(round(pos)), include_p=include_p)
        step = base_rr * (1.0 + rng.normal(0.0, cv))
        pos += max(step, SAMPLE_RATE * 0.3)
    return _zscore_12lead(lead)


class TestRateCategory:
    def test_normal_sinus_rhythm(self) -> None:
        r = analyze_rhythm(_regular_strip(bpm=72.0))
        assert r.rhythm_basis == "sinus"
        assert r.rate_category == "normal"
        assert r.classification == "Normal sinus rhythm"
        assert r.heart_rate_bpm is not None and abs(r.heart_rate_bpm - 72.0) < 6.0

    def test_sinus_bradycardia(self) -> None:
        r = analyze_rhythm(_regular_strip(bpm=48.0))
        assert r.rate_category == "bradycardia"
        assert r.classification == "Sinus bradycardia"

    def test_sinus_tachycardia(self) -> None:
        r = analyze_rhythm(_regular_strip(bpm=120.0))
        assert r.rate_category == "tachycardia"
        assert r.classification == "Sinus tachycardia"


class TestAtrialFibrillation:
    def test_irregular_no_p_is_af(self) -> None:
        r = analyze_rhythm(_irregular_strip(mean_bpm=95.0, include_p=False, cv=0.28))
        assert r.rhythm_basis == "atrial_fibrillation"
        assert not r.regular
        assert r.p_wave_fraction < 0.35
        assert "Atrial fibrillation" in r.classification

    def test_irregular_with_p_is_not_af(self) -> None:
        # Irregularity with preserved P waves must NOT be called AF.
        r = analyze_rhythm(_irregular_strip(mean_bpm=80.0, include_p=True, cv=0.22))
        assert r.rhythm_basis != "atrial_fibrillation"


class TestEctopy:
    def test_detects_premature_wide_beat(self) -> None:
        # Regular sinus at 72 bpm with one early, wide ventricular beat inserted.
        n_samples = SAMPLE_RATE * 10
        lead = np.zeros(n_samples, dtype=np.float64)
        rr = (60.0 / 72.0) * SAMPLE_RATE
        positions = [rr * i for i in range(1, 13)]
        # Make beat #6 premature (arrives at 0.6 RR after the prior beat) and wide.
        pvc_pos = positions[4] + rr * 0.6
        positions = positions[:5] + [pvc_pos] + positions[5:]
        for idx, pos in enumerate(sorted(positions)):
            if abs(pos - pvc_pos) < 1.0:
                _add_beat(lead, int(round(pos)), qrs_ms=160.0, qrs_amp=2.0, include_p=False)
            else:
                _add_beat(lead, int(round(pos)), include_p=True)
        r = analyze_rhythm(_zscore_12lead(lead))
        assert r.pvc_count >= 1
        assert r.ectopy_present

    def test_clean_sinus_has_no_ectopy(self) -> None:
        r = analyze_rhythm(_regular_strip(bpm=72.0))
        assert r.pvc_count == 0
        assert not r.ectopy_present


class TestRhythmStripPreference:
    def test_prefers_rhythm_strip_over_tiled_signal(self) -> None:
        tiled = _regular_strip(bpm=120.0)
        strip = _regular_strip(bpm=50.0)
        r = analyze_rhythm(tiled, rhythm_strip=strip)
        assert r.heart_rate_bpm is not None and abs(r.heart_rate_bpm - 50.0) < 6.0


class TestUnmeasurable:
    def test_flat_signal_is_undetermined(self) -> None:
        flat = np.zeros((12, SAMPLE_RATE * 10), dtype=np.float32)
        r = analyze_rhythm(flat)
        assert r.quality == "unmeasurable"
        assert r.rhythm_basis == "undetermined"
        assert r.heart_rate_bpm is None
