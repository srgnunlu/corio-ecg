# Tests for building a PTB-XL development manifest for VT/SVT criteria audit.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.build_vtsvt_ptbxl_manifest import (
    VTSVTPTBXLManifestConfig,
    build_ptbxl_vtsvt_manifest,
    select_ptbxl_vtsvt_records,
    write_manifest,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ecg_id": 1,
                "filename_hr": "records500/00000/00001_hr",
                "strat_fold": 10,
                "scp_codes": {"PSVT": 100.0, "CRBBB": 100.0},
            },
            {
                "ecg_id": 2,
                "filename_hr": "records500/00000/00002_hr",
                "strat_fold": 10,
                "scp_codes": {"SVTAC": 100.0},
            },
            {
                "ecg_id": 3,
                "filename_hr": "records500/00000/00003_hr",
                "strat_fold": 10,
                "scp_codes": {"STACH": 100.0, "CLBBB": 100.0},
            },
            {
                "ecg_id": 4,
                "filename_hr": "records500/00000/00004_hr",
                "strat_fold": 10,
                "scp_codes": {"PACE": 100.0},
            },
            {
                "ecg_id": 5,
                "filename_hr": "records500/00000/00005_hr",
                "strat_fold": 9,
                "scp_codes": {"IVCD": 100.0},
            },
            {
                "ecg_id": 6,
                "filename_hr": "records500/00000/00006_hr",
                "strat_fold": 10,
                "scp_codes": {"CLBBB": 100.0},
            },
        ]
    ).set_index("ecg_id", drop=False)


def test_select_ptbxl_vtsvt_records_is_deterministic_and_fold_scoped() -> None:
    config = VTSVTPTBXLManifestConfig(seed=7, per_category_limit=2, strat_fold=10)

    first = select_ptbxl_vtsvt_records(_metadata(), config)
    second = select_ptbxl_vtsvt_records(_metadata().sample(frac=1), config)

    assert [record["record_id"] for record in first] == [
        record["record_id"] for record in second
    ]
    assert {record["record_id"] for record in first} == {
        "ptbxl-1",
        "ptbxl-2",
        "ptbxl-3",
        "ptbxl-4",
        "ptbxl-6",
    }
    assert "ptbxl-5" not in {record["record_id"] for record in first}


def test_build_manifest_marks_ptbxl_as_negative_control_only() -> None:
    config = VTSVTPTBXLManifestConfig(seed=7, per_category_limit=2, strat_fold=10)

    manifest = build_ptbxl_vtsvt_manifest(_metadata(), Path("data/raw/ptb-xl"), config)

    assert manifest["version"] == "vtsvt-ptbxl-audit-manifest-v1"
    assert manifest["evidence_status"] == "development-negative-control"
    assert manifest["summary"]["selected_records"] == 5
    assert all(record["expected_supports_vt"] is False for record in manifest["records"])
    assert any(
        record["clinical_label"] == "SVT with bundle branch block"
        for record in manifest["records"]
    )
    assert manifest["records"][0]["record_path"].startswith("data/raw/ptb-xl/")


def test_write_manifest_refuses_accidental_overwrite(tmp_path: Path) -> None:
    manifest = build_ptbxl_vtsvt_manifest(
        _metadata(),
        Path("data/raw/ptb-xl"),
        VTSVTPTBXLManifestConfig(seed=7, per_category_limit=1, strat_fold=10),
    )
    output_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, output_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_manifest(manifest, output_path)


def test_write_manifest_outputs_valid_yaml(tmp_path: Path) -> None:
    manifest = build_ptbxl_vtsvt_manifest(
        _metadata(),
        Path("data/raw/ptb-xl"),
        VTSVTPTBXLManifestConfig(seed=7, per_category_limit=1, strat_fold=10),
    )
    output_path = tmp_path / "manifest.yaml"

    write_manifest(manifest, output_path)

    loaded = yaml.safe_load(output_path.read_text())
    assert loaded["version"] == "vtsvt-ptbxl-audit-manifest-v1"
    assert loaded["records"]
