"""Typed runtime quality decisions and stable audit reason codes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class QualityGateOutcome(StrEnum):
    """Possible runtime actions after digitization quality assessment."""

    ACCEPT = "accept"
    WARN = "warn"
    REJECT = "reject"


class QualityReasonCode(StrEnum):
    """Stable reason codes for UI rendering, audit, and analytics."""

    DIGITIZATION_FAILED = "digitization_failed"
    INVALID_FEATURES = "invalid_features"
    INSUFFICIENT_ACTIVE_LEADS = "insufficient_active_leads"
    INSUFFICIENT_LEAD_LABELS = "insufficient_lead_labels"
    LOW_EINTHOVEN_CONSISTENCY = "low_einthoven_consistency"
    UNCERTAIN_LAYOUT = "uncertain_layout"
    LOW_GRID_DENSITY = "low_grid_density"


@dataclass(frozen=True)
class QualityReason:
    """Stable reason code plus human-readable detail."""

    code: QualityReasonCode
    message: str


@dataclass(frozen=True)
class QualityGateDecision:
    """Runtime quality-gate outcome and auditable reasons."""

    outcome: QualityGateOutcome
    reason_details: tuple[QualityReason, ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return human-readable reason messages for compatibility."""
        return tuple(reason.message for reason in self.reason_details)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return stable reason-code values for audit and UI logic."""
        return tuple(reason.code.value for reason in self.reason_details)
