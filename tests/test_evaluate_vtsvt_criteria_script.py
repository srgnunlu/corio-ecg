# Tests for the VT/SVT deterministic criteria audit runner.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_vtsvt_criteria import (
    LoadedSignal,
    build_audit_report,
    evaluate_manifest,
    load_manifest,
    load_signal_record,
    write_report,
)
from src.vtsvt.models import (
    BrugadaCriteriaResult,
    LeadMorphology,
    VereckeiCriteriaResult,
    VTSVTAssessment,
    WCTFeatures,
)


def _lead(lead: str, *, pattern: str | None = "rs") -> LeadMorphology:
    return LeadMorphology(
        lead=lead,
        qrs_duration_ms=154.0,
        has_rs_complex=True,
        rs_interval_ms=84.0,
        initial_deflection="r",
        initial_deflection_width_ms=24.0,
        dominant_polarity="positive",
        vi_vt_ratio=1.4,
        initial_downstroke_notched=False,
        analyzed_beats=6,
        qrs_pattern=pattern,
        r_s_ratio=0.8,
        qrs_onset_to_s_nadir_ms=72.0,
        initial_r_taller_than_terminal_r=None,
        s_downstroke_notched=False,
    )


def _assessment(
    *,
    classification: str,
    supports_vt: bool,
    evidence: tuple[str, ...],
    terminal: bool | None = False,
) -> VTSVTAssessment:
    features = WCTFeatures(
        heart_rate_bpm=132.0,
        qrs_ms=156.0,
        regular=True,
        n_beats=12,
        anchor_lead="II",
        precordial_leads=(_lead("V1", pattern="r"), _lead("V6", pattern="rs")),
        avr=_lead("aVR", pattern="qr"),
        max_precordial_rs_interval_ms=84.0,
    )
    brugada = BrugadaCriteriaResult(
        supports_vt=supports_vt,
        positive_criteria=tuple(item for item in evidence if item.startswith("brugada_")),
        rs_absent_all_precordial=False,
        max_rs_interval_ms=84.0,
        av_dissociation_present=None,
        capture_or_fusion_beats_present=None,
        terminal_morphology_suggests_vt=terminal,
        terminal_morphology_criteria=tuple(
            item for item in evidence if item.startswith("brugada_terminal_")
        ),
    )
    vereckei = VereckeiCriteriaResult(
        supports_vt=any(item.startswith("vereckei_") for item in evidence),
        positive_criteria=tuple(item for item in evidence if item.startswith("vereckei_")),
        initial_r_in_avr=False,
        initial_r_or_q_width_gt_40ms=False,
        initial_downstroke_notched=False,
        vi_vt_ratio_leq_1=False,
        vi_vt_ratio=1.4,
    )
    return VTSVTAssessment(
        classification=classification,
        in_scope=True,
        supports_vt=supports_vt,
        supports_svt=False,
        evidence=evidence,
        limitations=("av_dissociation_not_assessed",),
        features=features,
        brugada=brugada,
        vereckei=vereckei,
    )


def test_load_manifest_reads_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "version: vtsvt-audit-manifest-v1\n"
        "sample_rate: 500\n"
        "records:\n"
        "  - record_id: vt-1\n"
        "    clinical_label: VT\n"
        "    signal_path: vt.npy\n"
    )

    manifest = load_manifest(manifest_path)

    assert manifest["version"] == "vtsvt-audit-manifest-v1"
    assert manifest["records"][0]["record_id"] == "vt-1"


def test_load_signal_record_accepts_repository_npy_shape(tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.npy"
    np.save(signal_path, np.ones((12, 5000), dtype=np.float32))
    record = {"record_id": "record-1", "signal_path": str(signal_path)}

    loaded = load_signal_record(record, tmp_path, default_sample_rate=500)

    assert loaded.signal.shape == (12, 5000)
    assert loaded.sample_rate == 500
    assert loaded.rhythm_strip is None


def test_load_signal_record_rejects_bad_signal_shape(tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.npy"
    np.save(signal_path, np.ones((4, 5000), dtype=np.float32))

    with pytest.raises(ValueError, match="signal must have shape"):
        load_signal_record({"signal_path": str(signal_path)}, tmp_path, default_sample_rate=500)


def test_evaluate_manifest_aggregates_criteria_and_outcomes(tmp_path: Path) -> None:
    manifest = {
        "version": "vtsvt-audit-manifest-v1",
        "sample_rate": 500,
        "records": [
            {
                "record_id": "vt-1",
                "clinical_label": "VT",
                "expected_supports_vt": True,
                "signal_path": "vt.npy",
            },
            {
                "record_id": "svt-1",
                "clinical_label": "SVT with aberrancy",
                "expected_supports_vt": False,
                "signal_path": "svt.npy",
            },
        ],
    }
    assessments = iter(
        [
            _assessment(
                classification="vt_supported",
                supports_vt=True,
                evidence=("brugada_terminal_v1_v6_rbbb_morphology",),
                terminal=True,
            ),
            _assessment(
                classification="indeterminate_wide_complex_tachycardia",
                supports_vt=False,
                evidence=(),
            ),
        ]
    )

    def signal_loader(
        record: dict[str, object],
        manifest_dir: Path,
        default_sample_rate: int,
    ) -> LoadedSignal:
        return LoadedSignal(np.zeros((12, 5000)), None, default_sample_rate, "test")

    def assess_fn(
        signal: np.ndarray,
        rhythm_strip: np.ndarray | None,
        sample_rate: int,
    ) -> VTSVTAssessment:
        return next(assessments)

    report = evaluate_manifest(
        manifest,
        tmp_path / "manifest.yaml",
        signal_loader=signal_loader,
        assess_fn=assess_fn,
    )

    assert report["aggregate"]["total"] == 2
    assert report["aggregate"]["successful"] == 2
    assert report["aggregate"]["confusion"]["true_positive"] == 1
    assert report["aggregate"]["confusion"]["true_negative"] == 1
    assert report["aggregate"]["criterion_counts"][
        "brugada_terminal_v1_v6_rbbb_morphology"
    ] == 1
    assert report["records"][0]["v1_qrs_pattern"] == "r"
    assert report["records"][0]["terminal_morphology_suggests_vt"] is True


def test_build_audit_report_marks_record_failures() -> None:
    report = build_audit_report(
        manifest={"version": "vtsvt-audit-manifest-v1", "records": []},
        manifest_path=Path("manifest.yaml"),
        records=[
            {
                "record_id": "bad",
                "status": "failed",
                "clinical_label": "VT",
                "expected_supports_vt": True,
                "error": "ValueError: bad signal",
            }
        ],
    )

    assert report["aggregate"]["failed"] == 1
    assert report["aggregate"]["confusion"]["evaluated"] == 0


def test_write_report_refuses_accidental_overwrite(tmp_path: Path) -> None:
    report = build_audit_report(
        manifest={"version": "vtsvt-audit-manifest-v1", "records": []},
        manifest_path=Path("manifest.yaml"),
        records=[],
    )
    write_report(report, tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_report(report, tmp_path)


def test_write_report_outputs_json_and_csv_with_lf(tmp_path: Path) -> None:
    report = build_audit_report(
        manifest={"version": "vtsvt-audit-manifest-v1", "records": []},
        manifest_path=Path("manifest.yaml"),
        records=[],
    )

    json_path, csv_path = write_report(report, tmp_path, allow_overwrite=True)

    assert json_path.exists()
    assert csv_path.exists()
    assert b"\r\n" not in csv_path.read_bytes()
