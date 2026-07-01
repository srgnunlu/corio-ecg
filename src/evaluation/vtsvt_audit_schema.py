# Shared schema for deterministic VT/SVT criteria audit artifacts.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CSV_FIELDS: tuple[str, ...] = (
    "record_id",
    "clinical_label",
    "expected_supports_vt",
    "source_path",
    "status",
    "error",
    "classification",
    "in_scope",
    "supports_vt",
    "heart_rate_bpm",
    "qrs_ms",
    "regular",
    "n_beats",
    "anchor_lead",
    "max_precordial_rs_interval_ms",
    "terminal_morphology_suggests_vt",
    "evidence",
    "limitations",
    "brugada_criteria",
    "vereckei_criteria",
    "v1_qrs_pattern",
    "v1_r_s_ratio",
    "v1_qrs_onset_to_s_nadir_ms",
    "v1_initial_deflection_width_ms",
    "v1_s_downstroke_notched",
    "v6_qrs_pattern",
    "v6_r_s_ratio",
    "v6_qrs_onset_to_s_nadir_ms",
    "v6_initial_deflection_width_ms",
    "v6_s_downstroke_notched",
)


@dataclass(frozen=True)
class LoadedSignal:
    """Signal loaded from a VTSVT audit manifest record."""

    signal: np.ndarray
    rhythm_strip: np.ndarray | None
    sample_rate: int
    source_path: str
