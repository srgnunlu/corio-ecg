"""Load locked tolerances for controlled digitization experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIGITIZATION_REGRESSION_CONFIG = (
    PROJECT_ROOT / "configs" / "digitization_regression_v1.yaml"
)


@dataclass(frozen=True)
class RegressionTolerances:
    """Maximum supported-category regressions allowed for promotion."""

    median_correlation_drop: float
    median_rmse_mv_increase: float
    median_snr_db_drop: float
    median_gain_error_increase: float
    failure_rate_increase: float
    median_runtime_increase_fraction: float
    quality_gate_reject_rate_increase: float

    def validate(self) -> None:
        """Reject negative tolerances."""
        if any(value < 0.0 for value in asdict(self).values()):
            raise ValueError("digitization regression tolerances must be non-negative")


@dataclass(frozen=True)
class DigitizationRegressionConfig:
    """Versioned category roles and promotion tolerances."""

    version: str
    supported_categories: tuple[str, ...]
    target_categories: tuple[str, ...]
    tolerances: RegressionTolerances

    def validate(self) -> None:
        """Validate category roles and tolerances."""
        if not self.version.strip():
            raise ValueError("digitization regression config version must not be empty")
        if not self.supported_categories:
            raise ValueError("at least one supported category is required")
        overlap = set(self.supported_categories) & set(self.target_categories)
        if overlap:
            raise ValueError(f"category roles overlap: {sorted(overlap)}")
        self.tolerances.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""
        return asdict(self)


def load_digitization_regression_config(
    path: Path = DEFAULT_DIGITIZATION_REGRESSION_CONFIG,
) -> DigitizationRegressionConfig:
    """Load and validate a digitization regression YAML configuration."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("digitization regression config must contain a mapping")
    try:
        config = DigitizationRegressionConfig(
            version=str(payload["version"]),
            supported_categories=tuple(payload["supported_categories"]),
            target_categories=tuple(payload["target_categories"]),
            tolerances=RegressionTolerances(**payload["tolerances"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid digitization regression config structure: {exc}") from exc
    config.validate()
    return config
