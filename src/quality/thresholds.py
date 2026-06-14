"""Load inference-only thresholds for the runtime quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_GATE_CONFIG_PATH = PROJECT_ROOT / "configs" / "quality_gate_v1.yaml"


@dataclass(frozen=True)
class InferenceThresholds:
    """Thresholds for quality features available during ordinary inference."""

    reject_active_leads_below: int
    reject_detected_leads_below: int
    severe_einthoven_below: float
    warn_einthoven_below: float
    layout_cost_above: float
    severe_detected_leads_below: int
    warn_detected_leads_below: int
    pixel_per_mm_below: float
    expected_active_leads: int
    severe_flags_to_reject: int

    def validate(self) -> None:
        """Validate ordering and ranges for inference-time thresholds."""
        if self.reject_active_leads_below > self.expected_active_leads:
            raise ValueError("reject active-lead threshold exceeds expected active leads")
        if not (
            self.reject_detected_leads_below
            <= self.severe_detected_leads_below
            <= self.warn_detected_leads_below
            <= self.expected_active_leads
        ):
            raise ValueError("detected-lead thresholds are not ordered")
        if self.severe_einthoven_below >= self.warn_einthoven_below:
            raise ValueError("severe Einthoven threshold must be below warn threshold")
        if self.severe_flags_to_reject < 1:
            raise ValueError("severe_flags_to_reject must be positive")


def load_inference_thresholds(
    path: Path = DEFAULT_QUALITY_GATE_CONFIG_PATH,
) -> InferenceThresholds:
    """Load and validate only the inference section of a quality-gate config."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("quality-gate config must contain a mapping")
    try:
        thresholds = InferenceThresholds(**payload["inference"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid inference threshold structure: {exc}") from exc
    thresholds.validate()
    return thresholds


DEFAULT_INFERENCE_THRESHOLDS = load_inference_thresholds()
