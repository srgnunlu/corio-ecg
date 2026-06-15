"""Evaluate a conservative layout-scope policy on tune-only gate evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.evaluation.quality_gate_metrics import evaluate_quality_gate_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT_SCOPE_CONFIG = PROJECT_ROOT / "configs" / "layout_scope_experiment_v1.yaml"


@dataclass(frozen=True)
class LayoutScopeConfig:
    """Research-only supported-layout policy."""

    version: str
    supported_layouts: tuple[str, ...]
    require_layout_hint: bool

    def validate(self) -> None:
        """Validate the research policy."""
        if not self.version.strip() or not self.supported_layouts:
            raise ValueError("layout scope requires a version and supported layouts")
        if len(set(self.supported_layouts)) != len(self.supported_layouts):
            raise ValueError("supported layouts must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy."""
        return asdict(self)


def load_layout_scope_config(
    path: Path = DEFAULT_LAYOUT_SCOPE_CONFIG,
) -> LayoutScopeConfig:
    """Load the research-only layout scope policy."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("layout scope config must contain a mapping")
    try:
        payload["supported_layouts"] = tuple(payload["supported_layouts"])
        config = LayoutScopeConfig(**payload)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid layout scope config structure: {exc}") from exc
    config.validate()
    return config


def evaluate_layout_scope(
    quality_report: dict[str, Any],
    fidelity_report: dict[str, Any],
    config: LayoutScopeConfig,
) -> dict[str, Any]:
    """Apply scope rejection before the existing gate outcome."""
    config.validate()
    if quality_report.get("selected_split") != "tune":
        raise ValueError("layout scope experiment may run only on the tune split")
    layouts = {
        (str(record["ecg_id"]), str(record["image_id"]), str(record["category"])): str(
            record.get("layout", "")
        )
        for record in fidelity_report["records"]
    }
    rows: list[dict[str, Any]] = []
    unsupported = 0
    for record in quality_report["records"]:
        key = (
            str(record["ecg_id"]),
            str(record["image_id"]),
            str(record["category"]),
        )
        if key not in layouts:
            raise ValueError(f"quality record is missing from fidelity report: {key}")
        layout = layouts[key]
        in_scope = bool(layout) and layout in config.supported_layouts
        prediction = record["prediction"] if in_scope else "reject"
        unsupported += int(not in_scope)
        rows.append(
            {
                **record,
                "layout": layout,
                "in_scope": in_scope,
                "base_prediction": record["prediction"],
                "prediction": prediction,
            }
        )
    return {
        "config": config.to_dict(),
        "evaluation_status": "tune-only-research",
        "limitations": (
            "This policy narrows supported layouts using tune data. It does not detect "
            "all low-fidelity records within supported layouts and is not external evidence."
        ),
        "unsupported_records_rejected": unsupported,
        "aggregate": evaluate_quality_gate_rows(rows),
        "records": rows,
    }


__all__ = ["evaluate_layout_scope", "load_layout_scope_config"]
