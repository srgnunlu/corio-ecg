"""Create deterministic benchmark splits while keeping ECG variants together."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUPED_SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "quality_gate_split_v1.yaml"
SPLIT_NAMES = ("train", "tune", "test")


@dataclass(frozen=True)
class GroupedSplitConfig:
    """Versioned grouped-split configuration."""

    version: str
    seed: int
    train_fraction: float
    tune_fraction: float
    test_fraction: float
    evidence_status: str = "retroactive-development-only"
    group_key: str = "ecg_id"
    category_key: str = "category"

    @property
    def fractions(self) -> dict[str, float]:
        """Return split fractions keyed by split name."""
        return {
            "train": self.train_fraction,
            "tune": self.tune_fraction,
            "test": self.test_fraction,
        }

    def validate(self) -> None:
        """Validate the grouped-split contract."""
        if not self.version.strip():
            raise ValueError("split config version must not be empty")
        if self.evidence_status not in {"retroactive-development-only", "pre-registered"}:
            raise ValueError("unsupported split evidence_status")
        if not math.isclose(sum(self.fractions.values()), 1.0, abs_tol=1e-9):
            raise ValueError("split fractions must sum to 1.0")
        if any(fraction <= 0.0 for fraction in self.fractions.values()):
            raise ValueError("split fractions must be positive")
        if not self.group_key or not self.category_key:
            raise ValueError("group_key and category_key must not be empty")


def load_grouped_split_config(
    path: Path = DEFAULT_GROUPED_SPLIT_CONFIG_PATH,
) -> GroupedSplitConfig:
    """Load and validate a grouped-split YAML configuration."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("grouped-split config must contain a mapping")
    try:
        config = GroupedSplitConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid grouped-split config structure: {exc}") from exc
    config.validate()
    return config


def _allocate_group_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw_counts = {name: total * fractions[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(raw_counts[name]) for name in SPLIT_NAMES}
    remaining = total - sum(counts.values())
    ranked = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw_counts[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def _validate_records(records: list[dict[str, Any]], config: GroupedSplitConfig) -> None:
    if not records:
        raise ValueError("cannot create grouped split from empty records")
    for index, record in enumerate(records):
        if not record.get(config.group_key):
            raise ValueError(f"record {index} is missing group key {config.group_key!r}")
        if not record.get(config.category_key):
            raise ValueError(f"record {index} is missing category key {config.category_key!r}")


def _assign_groups(
    grouped_records: dict[str, list[dict[str, Any]]],
    config: GroupedSplitConfig,
) -> dict[str, str]:
    group_counts = _allocate_group_counts(len(grouped_records), config.fractions)
    total_categories = Counter(
        str(record[config.category_key])
        for records in grouped_records.values()
        for record in records
    )
    target_categories = {
        split: {
            category: count * config.fractions[split]
            for category, count in total_categories.items()
        }
        for split in SPLIT_NAMES
    }
    assigned_categories: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_NAMES
    }
    remaining_slots = dict(group_counts)
    group_ids = sorted(grouped_records)
    random.Random(config.seed).shuffle(group_ids)
    assignments: dict[str, str] = {}

    for group_id in group_ids:
        categories = Counter(
            str(record[config.category_key]) for record in grouped_records[group_id]
        )
        candidates = [split for split in SPLIT_NAMES if remaining_slots[split] > 0]

        def score(split: str) -> tuple[float, int]:
            imbalance = sum(
                (
                    assigned_categories[split][category]
                    + categories[category]
                    - target_categories[split][category]
                )
                ** 2
                for category in total_categories
            )
            return imbalance, SPLIT_NAMES.index(split)

        selected = min(candidates, key=score)
        assignments[group_id] = selected
        assigned_categories[selected].update(categories)
        remaining_slots[selected] -= 1
    return assignments


def create_grouped_split_manifest(
    records: list[dict[str, Any]],
    config: GroupedSplitConfig,
) -> dict[str, Any]:
    """Create a deterministic category-aware split grouped by ECG identity."""
    config.validate()
    _validate_records(records, config)
    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[str(record[config.group_key])].append(record)
    assignments = _assign_groups(grouped_records, config)

    assigned_records = sorted(
        (
            {
                "ecg_id": str(record[config.group_key]),
                "image_id": record.get("image_id"),
                "category": str(record[config.category_key]),
                "split": assignments[str(record[config.group_key])],
            }
            for record in records
        ),
        key=lambda row: (
            str(row["ecg_id"]),
            str(row["category"]),
            str(row["image_id"]),
        ),
    )
    manifest = {
        "split_version": config.version,
        "seed": config.seed,
        "evidence_status": config.evidence_status,
        "group_key": config.group_key,
        "category_key": config.category_key,
        "fractions": config.fractions,
        "splits": {
            split: {
                "ecg_ids": sorted(
                    group_id for group_id, assigned in assignments.items() if assigned == split
                )
            }
            for split in SPLIT_NAMES
        },
        "summary": _build_summary(assigned_records),
        "records": assigned_records,
    }
    validate_grouped_split_manifest(manifest)
    return manifest


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    categories: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_NAMES
    }
    for record in records:
        split = str(record["split"])
        groups[split].add(str(record["ecg_id"]))
        categories[split][str(record["category"])] += 1
    return {
        "groups_total": len(set().union(*groups.values())),
        "records_total": len(records),
        "group_counts": {split: len(groups[split]) for split in SPLIT_NAMES},
        "record_counts": {
            split: sum(categories[split].values()) for split in SPLIT_NAMES
        },
        "category_counts": {
            split: dict(sorted(categories[split].items())) for split in SPLIT_NAMES
        },
    }


def validate_grouped_split_manifest(manifest: dict[str, Any]) -> None:
    """Reject manifests where one ECG identity crosses split boundaries."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in manifest.get("records", []):
        split = str(record.get("split"))
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}")
        group_splits[str(record.get("ecg_id"))].add(split)
    leaked = sorted(group_id for group_id, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"group leakage detected for ECG identities: {', '.join(leaked)}")

    listed_splits: dict[str, set[str]] = defaultdict(set)
    for split in SPLIT_NAMES:
        for group_id in manifest.get("splits", {}).get(split, {}).get("ecg_ids", []):
            listed_splits[str(group_id)].add(split)
    expected = {group_id: next(iter(splits)) for group_id, splits in group_splits.items()}
    listed = {
        group_id: next(iter(splits))
        for group_id, splits in listed_splits.items()
        if len(splits) == 1
    }
    if any(len(splits) != 1 for splits in listed_splits.values()) or listed != expected:
        raise ValueError("split ECG lists are inconsistent with record assignments")
