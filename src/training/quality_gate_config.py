"""Load and validate versioned quality-gate threshold configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.quality.thresholds import InferenceThresholds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_GATE_CONFIG_PATH = PROJECT_ROOT / "configs" / "quality_gate_v1.yaml"


@dataclass(frozen=True)
class FidelityThresholds:
    """Matched-reference thresholds used only for benchmark target labels."""

    reject_correlation_below: float
    warn_correlation_below: float
    reject_rmse_mv_above: float
    warn_rmse_mv_above: float
    reject_snr_db_below: float
    warn_snr_db_below: float

    def validate(self) -> None:
        """Validate that reject thresholds are stricter than warning thresholds."""
        if self.reject_correlation_below >= self.warn_correlation_below:
            raise ValueError("reject correlation threshold must be below warn correlation")
        if self.reject_rmse_mv_above <= self.warn_rmse_mv_above:
            raise ValueError("reject RMSE threshold must be above warn RMSE")
        if self.reject_snr_db_below >= self.warn_snr_db_below:
            raise ValueError("reject SNR threshold must be below warn SNR")


@dataclass(frozen=True)
class QualityGateConfig:
    """Complete versioned quality-gate configuration."""

    version: str
    fidelity: FidelityThresholds
    inference: InferenceThresholds

    def validate(self) -> None:
        """Validate the complete quality-gate configuration."""
        if not self.version.strip():
            raise ValueError("quality-gate config version must not be empty")
        self.fidelity.validate()
        self.inference.validate()

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved configuration for audit reports."""
        return asdict(self)


def load_quality_gate_config(
    path: Path = DEFAULT_QUALITY_GATE_CONFIG_PATH,
) -> QualityGateConfig:
    """Load and validate a quality-gate YAML configuration."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("quality-gate config must contain a mapping")
    try:
        config = QualityGateConfig(
            version=str(payload["version"]),
            fidelity=FidelityThresholds(**payload["fidelity"]),
            inference=InferenceThresholds(**payload["inference"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid quality-gate config structure: {exc}") from exc
    config.validate()
    return config
