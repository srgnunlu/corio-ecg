"""Tests for manifest-locked PMcardio holdout downloads."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.download_pmcardio_holdout import select_manifest_metadata


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ECG ID": "ecg-1",
                "Image relative path": "photos_iphone/img_1.jpeg",
            },
            {
                "ECG ID": "ecg-1",
                "Image relative path": "photos_scans/img_1.jpeg",
            },
            {
                "ECG ID": "ecg-2",
                "Image relative path": "photos_iphone/img_2.jpeg",
            },
        ]
    )


def test_select_manifest_metadata_adds_split_and_reference_key() -> None:
    manifest = {
        "records": [
            {"relative_path": "photos_iphone/img_1.jpeg", "split": "test"},
            {"relative_path": "photos_scans/img_1.jpeg", "split": "test"},
        ]
    }

    selected = select_manifest_metadata(_metadata(), manifest)

    assert len(selected) == 2
    assert set(selected["split"]) == {"test"}
    assert set(selected["reference_key"]) == {"ecg-1"}
    assert set(selected["category"]) == {"photos_iphone", "photos_scans"}


def test_select_manifest_metadata_rejects_missing_paths() -> None:
    manifest = {
        "records": [{"relative_path": "photos_iphone/missing.jpeg", "split": "test"}]
    }

    with pytest.raises(ValueError, match="metadata does not match"):
        select_manifest_metadata(_metadata(), manifest)


def test_select_manifest_metadata_rejects_duplicate_paths() -> None:
    record = {"relative_path": "photos_iphone/img_1.jpeg", "split": "test"}

    with pytest.raises(ValueError, match="duplicate image paths"):
        select_manifest_metadata(_metadata(), {"records": [record, record]})
