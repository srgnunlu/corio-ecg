"""Validate and extract quality features available during digitizer inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_FEATURE_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "quality_feature_contract_v1.yaml"
)


class QualityFeatureError(ValueError):
    """Raised when inference diagnostics violate the active feature contract."""


@dataclass(frozen=True)
class QualityFeatureDefinition:
    """Definition and validation rules for one inference-time feature."""

    value_type: str
    minimum: float | int
    maximum: float | int
    unit: str
    missing_policy: str
    description: str

    def validate(self, name: str) -> None:
        """Validate one feature definition."""
        if self.value_type not in {"float", "integer"}:
            raise ValueError(f"{name} has unsupported value_type {self.value_type!r}")
        if self.minimum > self.maximum:
            raise ValueError(f"{name} minimum exceeds maximum")
        if self.missing_policy != "reject":
            raise ValueError(f"{name} has unsupported missing_policy")
        if not self.unit or not self.description:
            raise ValueError(f"{name} must define unit and description")


@dataclass(frozen=True)
class QualityFeatureContract:
    """Versioned collection of active and planned quality features."""

    version: str
    features: dict[str, QualityFeatureDefinition]
    planned_unavailable_features: tuple[str, ...]

    def validate(self) -> None:
        """Validate the complete feature contract."""
        if not self.version.strip() or not self.features:
            raise ValueError("quality feature contract must have version and features")
        for name, definition in self.features.items():
            definition.validate(name)
        overlap = set(self.features) & set(self.planned_unavailable_features)
        if overlap:
            raise ValueError(f"planned unavailable features are active: {sorted(overlap)}")


@dataclass(frozen=True)
class QualityFeatures:
    """Typed features consumed by the interpretable quality gate."""

    layout_cost: float
    detected_leads_count: int
    nonzero_leads_count: int
    einthoven_score: float
    avg_pixel_per_mm: float
    raw_lines_count: int


def load_quality_feature_contract(
    path: Path = DEFAULT_QUALITY_FEATURE_CONTRACT_PATH,
) -> QualityFeatureContract:
    """Load and validate the active inference-time feature contract."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("quality feature contract must contain a mapping")
    try:
        definitions = {
            name: QualityFeatureDefinition(**definition)
            for name, definition in payload["features"].items()
        }
        contract = QualityFeatureContract(
            version=str(payload["version"]),
            features=definitions,
            planned_unavailable_features=tuple(
                str(name) for name in payload.get("planned_unavailable_features", [])
            ),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"invalid quality feature contract structure: {exc}") from exc
    contract.validate()
    return contract


DEFAULT_QUALITY_FEATURE_CONTRACT = load_quality_feature_contract()


def _validated_value(
    name: str,
    diagnostics: dict[str, Any],
    definition: QualityFeatureDefinition,
) -> float | int:
    if name not in diagnostics:
        raise QualityFeatureError(f"{name} is missing from inference diagnostics")
    value = diagnostics[name]
    if definition.value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise QualityFeatureError(f"{name} must be an integer")
        resolved: float | int = value
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QualityFeatureError(f"{name} must be numeric")
        resolved = float(value)
    if resolved < definition.minimum or resolved > definition.maximum:
        raise QualityFeatureError(
            f"{name} value {resolved} is outside "
            f"[{definition.minimum}, {definition.maximum}]"
        )
    return resolved


def extract_quality_features(
    diagnostics: dict[str, Any],
    contract: QualityFeatureContract = DEFAULT_QUALITY_FEATURE_CONTRACT,
) -> QualityFeatures:
    """Extract typed gate features or reject invalid inference diagnostics."""
    values = {
        name: _validated_value(name, diagnostics, definition)
        for name, definition in contract.features.items()
    }
    return QualityFeatures(
        layout_cost=float(values["layout_cost"]),
        detected_leads_count=int(values["detected_leads_count"]),
        nonzero_leads_count=int(values["nonzero_leads_count"]),
        einthoven_score=float(values["einthoven_score"]),
        avg_pixel_per_mm=float(values["avg_pixel_per_mm"]),
        raw_lines_count=int(values["raw_lines_count"]),
    )
