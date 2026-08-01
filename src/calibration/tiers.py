# Evidence tiers that separate trustworthy ECGFounder heads from unmeasurable ones.

from __future__ import annotations

from typing import Final

# A head needs this many positives on the calibration fold before its
# threshold and Platt fit mean anything. Below it, a "best" threshold is
# just noise fitted to a handful of records.
MINIMUM_CALIBRATION_POSITIVES: Final[int] = 10

# Discrimination and operating-point floors for promoting a head to
# `validated`. Chosen to match how the report layer speaks: a validated head
# is one we are willing to state as a finding without hedging.
VALIDATED_MINIMUM_AUROC: Final[float] = 0.80
VALIDATED_MINIMUM_F1: Final[float] = 0.30

TIER_VALIDATED: Final[str] = "validated"
TIER_PROVISIONAL: Final[str] = "provisional"
TIER_RESEARCH_ONLY: Final[str] = "research_only"

TIER_ORDER: Final[tuple[str, ...]] = (
    TIER_VALIDATED,
    TIER_PROVISIONAL,
    TIER_RESEARCH_ONLY,
)

TIER_DESCRIPTIONS: Final[dict[str, str]] = {
    TIER_VALIDATED: (
        "Sufficient calibration support and both AUROC and F1 above the "
        "promotion floors — reportable as a finding."
    ),
    TIER_PROVISIONAL: (
        "Measurable on the calibration fold but weak discrimination or a poor "
        "operating point — show with an explicit uncertainty caveat."
    ),
    TIER_RESEARCH_ONLY: (
        "Too few positives on the calibration fold to measure or calibrate — "
        "research output only, never a stated finding."
    ),
}


def assign_tier(positives: int, auroc: float | None, f1: float | None) -> str:
    """Classify one diagnosis head by how much evidence backs it.

    Args:
        positives: Positive examples for this head on the calibration fold.
        auroc: Calibration-fold AUROC, or None when it could not be computed.
        f1: Calibration-fold F1 at the selected threshold, or None.

    Returns:
        One of TIER_VALIDATED, TIER_PROVISIONAL, TIER_RESEARCH_ONLY.
    """
    if positives < MINIMUM_CALIBRATION_POSITIVES or auroc is None or f1 is None:
        return TIER_RESEARCH_ONLY
    if auroc >= VALIDATED_MINIMUM_AUROC and f1 >= VALIDATED_MINIMUM_F1:
        return TIER_VALIDATED
    return TIER_PROVISIONAL
