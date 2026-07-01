# Build PTB-XL negative-control manifests for VT/SVT criteria audits.

from __future__ import annotations

import ast
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]

SVT_CODES = frozenset({"SVTAC", "PSVT"})
TACHY_CODES = frozenset({"STACH"})
ATRIAL_CODES = frozenset({"AFIB", "AFLT"})
WIDE_QRS_CODES = frozenset({"CRBBB", "CLBBB", "IVCD"})
CONDUCTION_CODES = WIDE_QRS_CODES | frozenset({"IRBBB", "ILBBB", "WPW"})
PACED_CODES = frozenset({"PACE"})

CATEGORY_RULES: tuple[tuple[str, str, frozenset[str], frozenset[str]], ...] = (
    (
        "svt_with_bundle_branch_block",
        "SVT with bundle branch block",
        SVT_CODES,
        CONDUCTION_CODES,
    ),
    ("svt_or_psvt", "SVT or PSVT", SVT_CODES, frozenset()),
    (
        "sinus_tachycardia_with_wide_qrs",
        "Sinus tachycardia with wide-QRS conduction",
        TACHY_CODES,
        WIDE_QRS_CODES,
    ),
    ("paced_rhythm", "Paced rhythm", PACED_CODES, frozenset()),
    (
        "atrial_arrhythmia_with_wide_qrs",
        "Atrial arrhythmia with wide-QRS conduction",
        ATRIAL_CODES,
        WIDE_QRS_CODES,
    ),
    (
        "bundle_branch_block_or_ivcd",
        "Bundle branch block or IVCD",
        WIDE_QRS_CODES,
        frozenset(),
    ),
)


@dataclass(frozen=True)
class VTSVTPTBXLManifestConfig:
    """Selection configuration for the PTB-XL VTSVT development manifest."""

    seed: int = 20260620
    per_category_limit: int = 12
    strat_fold: int = 10
    sample_rate: int = 500
    version: str = "vtsvt-ptbxl-audit-manifest-v1"

    def validate(self) -> None:
        """Validate config values before selection."""
        if self.per_category_limit <= 0:
            raise ValueError("per_category_limit must be positive")
        if self.strat_fold < 1:
            raise ValueError("strat_fold must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")


def load_ptbxl_metadata(data_dir: Path) -> pd.DataFrame:
    """Load PTB-XL metadata with parsed SCP code dictionaries."""
    csv_path = data_dir / "ptbxl_database.csv"
    metadata = pd.read_csv(csv_path, index_col="ecg_id")
    metadata["ecg_id"] = metadata.index.astype(int)
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    return metadata


def select_ptbxl_vtsvt_records(
    metadata: pd.DataFrame,
    config: VTSVTPTBXLManifestConfig,
) -> list[dict[str, Any]]:
    """Select deterministic PTB-XL negative controls for VTSVT criteria audit."""
    config.validate()
    fold_records = metadata[metadata.strat_fold == config.strat_fold].copy()
    selected_ecg_ids: set[int] = set()
    selected: list[dict[str, Any]] = []

    for category, clinical_label, required, optional in CATEGORY_RULES:
        candidates = _matching_records(fold_records, required, optional)
        candidates = [
            row for row in candidates if int(row["ecg_id"]) not in selected_ecg_ids
        ]
        chosen = _sample_records(candidates, config.seed, category, config.per_category_limit)
        for row in chosen:
            ecg_id = int(row["ecg_id"])
            selected_ecg_ids.add(ecg_id)
            selected.append(_manifest_record(row, category, clinical_label))
    return selected


def build_ptbxl_vtsvt_manifest(
    metadata: pd.DataFrame,
    data_dir: Path,
    config: VTSVTPTBXLManifestConfig,
) -> dict[str, Any]:
    """Build a VTSVT audit manifest from PTB-XL metadata."""
    records = select_ptbxl_vtsvt_records(metadata, config)
    for record in records:
        filename = record.pop("_filename_hr")
        record["record_path"] = str(data_dir / str(filename))

    return {
        "version": config.version,
        "source_dataset": "PTB-XL",
        "evidence_status": "development-negative-control",
        "sample_rate": config.sample_rate,
        "selection": {
            "seed": config.seed,
            "strat_fold": config.strat_fold,
            "per_category_limit": config.per_category_limit,
        },
        "cohort_description": (
            "PTB-XL development negative-control cohort for VT/SVT criteria "
            "false-positive analysis. PTB-XL does not provide adjudicated VT "
            "positive labels in this builder."
        ),
        "summary": _summary(records),
        "records": records,
    }


def write_manifest(
    manifest: dict[str, Any],
    output_path: Path,
    *,
    allow_overwrite: bool = False,
) -> None:
    """Write a YAML manifest while protecting existing generated artifacts."""
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing VTSVT PTB-XL manifest: {output_path}. "
            "Use --allow-overwrite for intentional regeneration."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def _matching_records(
    metadata: pd.DataFrame,
    required_codes: frozenset[str],
    optional_codes: frozenset[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = metadata.reset_index(drop=True).sort_values("ecg_id")
    for row in ordered.to_dict("records"):
        codes = set(row["scp_codes"])
        if not codes.intersection(required_codes):
            continue
        if optional_codes and not codes.intersection(optional_codes):
            continue
        rows.append(row)
    return rows


def _sample_records(
    records: list[dict[str, Any]],
    seed: int,
    category: str,
    limit: int,
) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{category}")
    shuffled = list(records)
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda row: int(row["ecg_id"]))


def _manifest_record(
    row: dict[str, Any],
    category: str,
    clinical_label: str,
) -> dict[str, Any]:
    scp_codes = dict(sorted(row["scp_codes"].items()))
    return {
        "record_id": f"ptbxl-{int(row['ecg_id'])}",
        "clinical_label": clinical_label,
        "category": category,
        "expected_supports_vt": False,
        "source_scp_codes": scp_codes,
        "_filename_hr": str(row["filename_hr"]),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(record["category"]) for record in records)
    labels = Counter(str(record["clinical_label"]) for record in records)
    return {
        "selected_records": len(records),
        "category_counts": dict(sorted(categories.items())),
        "clinical_label_counts": dict(sorted(labels.items())),
        "expected_supports_vt_counts": {"false": len(records)},
    }
