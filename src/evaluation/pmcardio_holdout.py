"""Create a pre-registered PMcardio holdout from unused physical ECGs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PMCARDIO_HOLDOUT_CONFIG = PROJECT_ROOT / "configs" / "pmcardio_holdout_v1.yaml"


@dataclass(frozen=True)
class PMCardioHoldoutConfig:
    """Locked selection and split contract for the internal holdout."""

    version: str
    evidence_status: str
    seed: int
    tune_ecg_count: int
    test_ecg_count: int
    categories: tuple[str, ...]

    def validate(self) -> None:
        """Validate the holdout selection contract."""
        if not self.version.strip() or self.evidence_status != "pre-registered":
            raise ValueError("holdout must have a version and pre-registered status")
        if self.tune_ecg_count < 1 or self.test_ecg_count < 30:
            raise ValueError("holdout requires tune ECGs and at least 30 test ECGs")
        if len(self.categories) < 2 or len(set(self.categories)) != len(self.categories):
            raise ValueError("holdout categories must be unique and contain at least two values")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration."""
        return asdict(self)


def load_pmcardio_holdout_config(
    path: Path = DEFAULT_PMCARDIO_HOLDOUT_CONFIG,
) -> PMCardioHoldoutConfig:
    """Load and validate the pre-registered holdout configuration."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("PMcardio holdout config must contain a mapping")
    try:
        payload["categories"] = tuple(payload["categories"])
        config = PMCardioHoldoutConfig(**payload)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid PMcardio holdout config structure: {exc}") from exc
    config.validate()
    return config


def _category(relative_path: object) -> str:
    return str(relative_path).split("/", maxsplit=1)[0]


def _complete_physical_records(
    metadata: pd.DataFrame,
    categories: tuple[str, ...],
) -> pd.DataFrame:
    records = metadata.copy()
    records["category"] = records["Image relative path"].map(_category)
    records = records[
        records["category"].isin(categories) & (records["Image page"] == 0)
    ].copy()
    category_counts = records.groupby("ECG ID")["category"].nunique()
    complete_ids = set(category_counts[category_counts == len(categories)].index.astype(str))
    records = records[records["ECG ID"].astype(str).isin(complete_ids)].copy()
    duplicate_pairs = records.duplicated(["ECG ID", "category"], keep=False)
    if duplicate_pairs.any():
        raise ValueError("PMcardio metadata contains duplicate ECG/category records")
    return records


def _assign_splits(
    candidates: pd.DataFrame,
    config: PMCardioHoldoutConfig,
) -> dict[str, str]:
    layout_by_ecg = (
        candidates[["ECG ID", "ECG format"]]
        .drop_duplicates()
        .set_index("ECG ID")["ECG format"]
        .astype(str)
        .to_dict()
    )
    ecg_ids = sorted(str(value) for value in candidates["ECG ID"].unique())
    random.Random(config.seed).shuffle(ecg_ids)
    assignments: dict[str, str] = {}
    tune_layouts: Counter[str] = Counter()
    total_layouts = Counter(layout_by_ecg.values())
    tune_target = {
        layout: count * config.tune_ecg_count / len(ecg_ids)
        for layout, count in total_layouts.items()
    }
    for ecg_id in ecg_ids:
        remaining_tune = config.tune_ecg_count - sum(tune_layouts.values())
        remaining_groups = len(ecg_ids) - len(assignments)
        layout = layout_by_ecg[ecg_id]
        select_tune = remaining_tune > 0 and (
            remaining_tune == remaining_groups or tune_layouts[layout] < tune_target[layout]
        )
        assignments[ecg_id] = "tune" if select_tune else "test"
        if select_tune:
            tune_layouts[layout] += 1
    return assignments


def build_pmcardio_holdout_manifest(
    metadata: pd.DataFrame,
    development_manifest: dict[str, Any],
    config: PMCardioHoldoutConfig,
    *,
    metadata_sha256: str,
) -> dict[str, Any]:
    """Select unused ECGs and create a stable pre-registered holdout manifest."""
    config.validate()
    records = _complete_physical_records(metadata, config.categories)
    development_ids = {
        str(record["ECG ID"]) for record in development_manifest.get("records", [])
    }
    candidates = records[~records["ECG ID"].astype(str).isin(development_ids)].copy()
    expected_count = config.tune_ecg_count + config.test_ecg_count
    available_ids = set(candidates["ECG ID"].astype(str))
    if len(available_ids) != expected_count:
        raise ValueError(
            f"holdout contract requires exactly {expected_count} unused complete ECGs; "
            f"found {len(available_ids)}"
        )
    assignments = _assign_splits(candidates, config)
    output_records = [
        {
            "ecg_id": str(row["ECG ID"]),
            "image_id": int(row["Image ID"]),
            "category": str(row["category"]),
            "layout": str(row["ECG format"]),
            "relative_path": str(row["Image relative path"]),
            "split": assignments[str(row["ECG ID"])],
        }
        for _, row in candidates.sort_values(["ECG ID", "category"]).iterrows()
    ]
    split_counts = Counter(assignments.values())
    layout_counts = {
        split: dict(
            sorted(
                Counter(
                    record["layout"]
                    for record in output_records
                    if record["split"] == split
                    and record["category"] == config.categories[0]
                ).items()
            )
        )
        for split in ("tune", "test")
    }
    manifest = {
        "manifest_version": config.version,
        "evidence_status": config.evidence_status,
        "seed": config.seed,
        "group_key": "ecg_id",
        "category_key": "category",
        "metadata_sha256": metadata_sha256,
        "excluded_development_manifest_id": development_manifest.get("manifest_id"),
        "config": config.to_dict(),
        "splits": {
            "train": {"ecg_ids": []},
            "tune": {
                "ecg_ids": sorted(
                    ecg_id for ecg_id, split in assignments.items() if split == "tune"
                )
            },
            "test": {
                "ecg_ids": sorted(
                    ecg_id for ecg_id, split in assignments.items() if split == "test"
                )
            },
        },
        "summary": {
            "ecg_groups": len(assignments),
            "images": len(output_records),
            "split_ecg_counts": dict(sorted(split_counts.items())),
            "split_layout_counts": layout_counts,
        },
        "records": output_records,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_id"] = hashlib.sha256(canonical).hexdigest()
    return manifest
