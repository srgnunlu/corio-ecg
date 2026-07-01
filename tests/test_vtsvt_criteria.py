# Tests for deterministic VT/SVT wide-complex tachycardia criteria features.

import numpy as np

from src.measurement.intervals import IntervalMeasurements
from src.measurement.rhythm_analysis import RhythmAnalysis
from src.vtsvt.criteria import assess_vtsvt
from src.vtsvt.features import extract_wct_features

SAMPLE_RATE = 500


def _ms(value: float) -> int:
    return int(round(value / 1000.0 * SAMPLE_RATE))


def _draw_polyline(lead: np.ndarray, qrs_on: int, points: list[tuple[float, float]]) -> None:
    """Draw a piecewise-linear QRS morphology from millisecond/amplitude points."""
    for (start_ms, start_amp), (end_ms, end_amp) in zip(points, points[1:]):
        start = qrs_on + _ms(start_ms)
        end = qrs_on + _ms(end_ms)
        if end <= start:
            continue
        lo, hi = max(start, 0), min(end, lead.size)
        if hi <= lo:
            continue
        fraction = (np.arange(lo, hi) - start) / (end - start)
        lead[lo:hi] += start_amp + fraction * (end_amp - start_amp)


def _signal_with_qrs(
    *,
    bpm: float = 130.0,
    precordial_points: list[tuple[float, float]],
    v1_points: list[tuple[float, float]] | None = None,
    v6_points: list[tuple[float, float]] | None = None,
    avr_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Build a 12-lead synthetic tachycardia with configurable lead morphologies."""
    n_samples = SAMPLE_RATE * 10
    signal = np.zeros((12, n_samples), dtype=np.float64)
    rr = 60.0 / bpm * SAMPLE_RATE
    qrs_on = rr
    lead_ii = [(0.0, 0.0), (30.0, 1.5), (150.0, -0.4), (170.0, 0.0)]
    avr_shape = avr_points or [(0.0, 0.0), (12.0, -1.0), (24.0, 0.0), (105.0, 0.0)]

    while qrs_on < n_samples - rr * 0.5:
        qrs_i = int(round(qrs_on))
        _draw_polyline(signal[1], qrs_i, lead_ii)
        _draw_polyline(signal[3], qrs_i, avr_shape)
        for lead_idx in range(6, 12):
            lead_points = precordial_points
            if lead_idx == 6 and v1_points is not None:
                lead_points = v1_points
            if lead_idx == 11 and v6_points is not None:
                lead_points = v6_points
            _draw_polyline(signal[lead_idx], qrs_i, lead_points)
        qrs_on += rr

    mean, std = float(np.mean(signal)), float(np.std(signal))
    return ((signal - mean) / (std + 1e-8)).astype(np.float32)


def _intervals(*, bpm: float = 130.0, qrs_ms: float = 160.0) -> IntervalMeasurements:
    return IntervalMeasurements(
        heart_rate_bpm=bpm,
        rr_ms=60000.0 / bpm,
        pr_ms=None,
        qrs_ms=qrs_ms,
        qt_ms=None,
        qtc_bazett_ms=None,
        qtc_fridericia_ms=None,
        qtc_preferred_ms=None,
        qtc_formula="n/a",
        measured_lead="II",
        n_beats=16,
        quality="good",
    )


def _rhythm(*, bpm: float = 130.0, regular: bool = True) -> RhythmAnalysis:
    return RhythmAnalysis(
        classification="Undetermined rhythm (tachycardic)",
        rhythm_basis="undetermined",
        rate_category="tachycardia",
        heart_rate_bpm=bpm,
        rr_mean_ms=60000.0 / bpm,
        rr_cv=0.02 if regular else 0.22,
        rr_rmssd_ms=10.0,
        regular=regular,
        p_wave_fraction=0.0,
        p_waves_present=False,
        pvc_count=0,
        ectopy_present=False,
        n_beats=16,
        measured_lead="II",
        quality="good",
    )


def test_brugada_long_rs_interval_supports_vt() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.2), (150.0, -1.0), (175.0, 0.0)]
    )

    result = assess_vtsvt(signal, intervals=_intervals(), rhythm=_rhythm())

    assert result.in_scope
    assert result.classification == "vt_supported"
    assert "brugada_rs_interval_gt_100ms" in result.evidence
    assert result.brugada.max_rs_interval_ms is not None
    assert result.brugada.max_rs_interval_ms > 100.0


def test_brugada_absent_rs_in_all_precordials_supports_vt() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (80.0, 1.2), (165.0, 0.0)]
    )

    result = assess_vtsvt(signal, intervals=_intervals(), rhythm=_rhythm())

    assert result.classification == "vt_supported"
    assert result.brugada.rs_absent_all_precordial
    assert "brugada_absent_rs_all_precordial" in result.evidence


def test_vereckei_avr_initial_r_is_audited() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.0), (85.0, -1.0), (135.0, 0.0)],
        avr_points=[(0.0, 0.0), (55.0, 1.1), (130.0, 0.3), (170.0, -0.2)],
    )

    result = assess_vtsvt(signal, intervals=_intervals(), rhythm=_rhythm())

    assert result.vereckei.supports_vt
    assert "vereckei_initial_r_in_avr" in result.evidence
    assert result.features.avr is not None
    assert result.features.avr.initial_deflection == "r"
    assert result.features.avr.initial_deflection_width_ms is not None
    assert result.features.avr.initial_deflection_width_ms > 40.0


def test_no_positive_criteria_stays_indeterminate_not_svt() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.2), (70.0, -0.5), (120.0, 0.0)]
    )

    result = assess_vtsvt(signal, intervals=_intervals(), rhythm=_rhythm())

    assert result.in_scope
    assert result.classification == "indeterminate_wide_complex_tachycardia"
    assert not result.supports_vt
    assert not result.supports_svt
    assert result.brugada.terminal_morphology_suggests_vt is False
    assert "terminal_v1_v6_morphology_criteria_not_yet_implemented" not in result.limitations


def test_extracts_terminal_v1_v6_morphology_features() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.0), (75.0, -1.0), (130.0, 0.0)],
        v1_points=[(0.0, 0.0), (85.0, 1.2), (165.0, 0.0)],
        v6_points=[(0.0, 0.0), (20.0, 0.35), (72.0, -1.1), (140.0, 0.0)],
    )

    features = extract_wct_features(signal, intervals=_intervals(), rhythm=_rhythm())
    v1 = next(lead for lead in features.precordial_leads if lead.lead == "V1")
    v6 = next(lead for lead in features.precordial_leads if lead.lead == "V6")

    assert v1.qrs_pattern == "r"
    assert v1.dominant_polarity == "positive"
    assert v6.qrs_pattern == "rs"
    assert v6.r_s_ratio is not None
    assert v6.r_s_ratio < 1.0


def test_brugada_terminal_rbbb_v1_v6_morphology_supports_vt() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.0), (75.0, -1.0), (130.0, 0.0)],
        v1_points=[(0.0, 0.0), (85.0, 1.2), (165.0, 0.0)],
        v6_points=[(0.0, 0.0), (20.0, 0.35), (72.0, -1.1), (140.0, 0.0)],
    )

    result = assess_vtsvt(signal, intervals=_intervals(), rhythm=_rhythm())

    assert result.classification == "vt_supported"
    assert result.brugada.terminal_morphology_suggests_vt is True
    assert "brugada_terminal_v1_v6_rbbb_morphology" in result.evidence
    assert "brugada_rs_interval_gt_100ms" not in result.evidence


def test_brugada_terminal_lbbb_v1_morphology_supports_vt() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.0), (75.0, -1.0), (130.0, 0.0)],
        v1_points=[(0.0, 0.0), (42.0, 0.45), (88.0, -1.2), (150.0, 0.0)],
        v6_points=[(0.0, 0.0), (25.0, 1.0), (120.0, 0.2), (150.0, 0.0)],
    )

    result = assess_vtsvt(signal, intervals=_intervals(), rhythm=_rhythm())
    v1 = next(lead for lead in result.features.precordial_leads if lead.lead == "V1")

    assert result.classification == "vt_supported"
    assert result.brugada.terminal_morphology_suggests_vt is True
    assert "brugada_terminal_v1_lbbb_morphology" in result.evidence
    assert v1.initial_deflection_width_ms is not None
    assert v1.initial_deflection_width_ms > 30.0
    assert v1.qrs_onset_to_s_nadir_ms is not None
    assert v1.qrs_onset_to_s_nadir_ms > 60.0


def test_external_av_dissociation_flag_is_audited() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.0), (75.0, -1.0), (130.0, 0.0)]
    )

    result = assess_vtsvt(
        signal,
        intervals=_intervals(),
        rhythm=_rhythm(),
        av_dissociation_present=True,
    )

    assert result.classification == "vt_supported"
    assert "brugada_av_dissociation" in result.evidence
    assert result.brugada.av_dissociation_present is True


def test_narrow_tachycardia_is_out_of_scope() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.0), (75.0, -1.0), (105.0, 0.0)]
    )

    result = assess_vtsvt(signal, intervals=_intervals(qrs_ms=90.0), rhythm=_rhythm())

    assert not result.in_scope
    assert result.classification == "not_wide_complex_tachycardia"
    assert result.evidence == ()


def test_extracts_max_precordial_rs_interval_feature() -> None:
    signal = _signal_with_qrs(
        precordial_points=[(0.0, 0.0), (20.0, 1.2), (150.0, -1.0), (175.0, 0.0)]
    )

    features = extract_wct_features(signal, intervals=_intervals(), rhythm=_rhythm())

    assert features.max_precordial_rs_interval_ms is not None
    assert features.max_precordial_rs_interval_ms > 100.0
    assert features.precordial_leads
