# Data models for deterministic VT/SVT wide-complex tachycardia assessment.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeadMorphology:
    """QRS morphology features extracted from one lead."""

    lead: str
    qrs_duration_ms: float | None
    has_rs_complex: bool
    rs_interval_ms: float | None
    initial_deflection: str | None  # "r" | "q"
    initial_deflection_width_ms: float | None
    dominant_polarity: str | None  # "positive" | "negative"
    vi_vt_ratio: float | None
    initial_downstroke_notched: bool
    analyzed_beats: int
    qrs_pattern: str | None = None
    r_s_ratio: float | None = None
    qrs_onset_to_s_nadir_ms: float | None = None
    initial_r_taller_than_terminal_r: bool | None = None
    s_downstroke_notched: bool = False


@dataclass(frozen=True)
class WCTFeatures:
    """Signal-derived features used by VT/SVT criteria engines."""

    heart_rate_bpm: float | None
    qrs_ms: float | None
    regular: bool | None
    n_beats: int
    anchor_lead: str | None
    precordial_leads: tuple[LeadMorphology, ...]
    avr: LeadMorphology | None
    max_precordial_rs_interval_ms: float | None


@dataclass(frozen=True)
class BrugadaCriteriaResult:
    """Auditable Brugada-style criteria result."""

    supports_vt: bool
    positive_criteria: tuple[str, ...]
    rs_absent_all_precordial: bool
    max_rs_interval_ms: float | None
    av_dissociation_present: bool | None
    capture_or_fusion_beats_present: bool | None
    terminal_morphology_suggests_vt: bool | None = None
    terminal_morphology_criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class VereckeiCriteriaResult:
    """Auditable aVR/Vereckei-style criteria result."""

    supports_vt: bool
    positive_criteria: tuple[str, ...]
    initial_r_in_avr: bool | None
    initial_r_or_q_width_gt_40ms: bool | None
    initial_downstroke_notched: bool | None
    vi_vt_ratio_leq_1: bool | None
    vi_vt_ratio: float | None


@dataclass(frozen=True)
class VTSVTAssessment:
    """Conservative VT/SVT assessment with evidence and limitations."""

    classification: str
    in_scope: bool
    supports_vt: bool
    supports_svt: bool
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    features: WCTFeatures
    brugada: BrugadaCriteriaResult
    vereckei: VereckeiCriteriaResult
