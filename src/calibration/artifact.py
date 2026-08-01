# Versioned per-class calibration artefact: Platt parameters, thresholds and evidence tiers.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.calibration.tiers import TIER_RESEARCH_ONLY, TIER_VALIDATED
from src.utils.ecg_labels import DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1

# Probabilities are clipped before the logit so a saturated 0.0/1.0 output
# cannot produce infinities in the Platt transform.
_PROBABILITY_EPSILON = 1e-6

DEFAULT_CALIBRATION_PATH = Path("configs/calibration/ptbxl_fold9_v1.json")
# Escape hatch mirroring CORIO_SEGMENT_ENSEMBLE: set to "0" to run the
# uncalibrated path without moving the artefact file.
CALIBRATION_ENV_FLAG = "CORIO_CALIBRATION"


@dataclass(frozen=True)
class ClassCalibration:
    """Calibration state for one of the 150 ECGFounder heads."""

    label: str
    index: int
    tier: str
    positives: int
    threshold: float
    # None means "leave the raw probability alone" — used when a head has too
    # little support for a Platt fit to be meaningful.
    platt_slope: float | None = None
    platt_intercept: float | None = None
    fit_auroc: float | None = None
    fit_f1: float | None = None
    fit_f1_at_default_threshold: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether a Platt transform was fitted for this head."""
        return self.platt_slope is not None and self.platt_intercept is not None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "label": self.label,
            "index": self.index,
            "tier": self.tier,
            "positives": self.positives,
            "threshold": self.threshold,
            "platt_slope": self.platt_slope,
            "platt_intercept": self.platt_intercept,
            "fit_auroc": self.fit_auroc,
            "fit_f1": self.fit_f1,
            "fit_f1_at_default_threshold": self.fit_f1_at_default_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ClassCalibration:
        """Rebuild from a serialized dictionary."""
        return cls(
            label=payload["label"],
            index=int(payload["index"]),
            tier=payload["tier"],
            positives=int(payload["positives"]),
            threshold=float(payload["threshold"]),
            platt_slope=payload.get("platt_slope"),
            platt_intercept=payload.get("platt_intercept"),
            fit_auroc=payload.get("fit_auroc"),
            fit_f1=payload.get("fit_f1"),
            fit_f1_at_default_threshold=payload.get("fit_f1_at_default_threshold"),
        )


@dataclass(frozen=True)
class CalibrationArtifact:
    """All per-class calibration state learned from one validation fold."""

    fold: int
    record_count: int
    ground_truth_source: str
    classes: dict[str, ClassCalibration] = field(default_factory=dict)
    default_threshold: float = DEFAULT_THRESHOLD
    version: int = ARTIFACT_VERSION

    def threshold_for(self, label: str) -> float:
        """Return the learned cutoff for a head, or the global default."""
        entry = self.classes.get(label)
        return entry.threshold if entry else self.default_threshold

    def tier_for(self, label: str) -> str:
        """Return the evidence tier for a head.

        Heads absent from the artefact were never measured, so they are
        research-only by construction.
        """
        entry = self.classes.get(label)
        return entry.tier if entry else TIER_RESEARCH_ONLY

    def labels_in_tier(self, tier: str) -> list[str]:
        """List head labels belonging to one evidence tier."""
        return [label for label, entry in self.classes.items() if entry.tier == tier]

    def apply(self, probabilities: np.ndarray, labels: list[str]) -> np.ndarray:
        """Map raw sigmoid outputs to calibrated probabilities.

        Platt scaling is monotonic, so ranking metrics (AUROC, average
        precision) are unchanged; only the probability scale moves.

        Args:
            probabilities: Raw probabilities, shape (..., n_classes).
            labels: Class names in column order.

        Returns:
            Calibrated probabilities with the same shape.
        """
        calibrated = np.array(probabilities, dtype=np.float64, copy=True)
        for column, label in enumerate(labels):
            entry = self.classes.get(label)
            if entry is None or not entry.is_calibrated:
                continue
            clipped = np.clip(
                calibrated[..., column], _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
            )
            logit = np.log(clipped / (1.0 - clipped))
            calibrated[..., column] = 1.0 / (
                1.0 + np.exp(-(entry.platt_slope * logit + entry.platt_intercept))
            )
        return calibrated

    def threshold_map(self) -> dict[str, float]:
        """Per-class thresholds keyed by label, for the metrics helpers."""
        return {label: entry.threshold for label, entry in self.classes.items()}

    def summary(self) -> dict:
        """Counts per evidence tier, for logging and reports."""
        counts: dict[str, int] = {}
        for entry in self.classes.values():
            counts[entry.tier] = counts.get(entry.tier, 0) + 1
        return {
            "version": self.version,
            "fold": self.fold,
            "record_count": self.record_count,
            "ground_truth_source": self.ground_truth_source,
            "classes_with_state": len(self.classes),
            "tier_counts": counts,
            "validated_labels": sorted(self.labels_in_tier(TIER_VALIDATED)),
        }

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "fold": self.fold,
            "record_count": self.record_count,
            "ground_truth_source": self.ground_truth_source,
            "default_threshold": self.default_threshold,
            "classes": [entry.to_dict() for entry in self.classes.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CalibrationArtifact:
        """Rebuild from a serialized dictionary."""
        version = int(payload.get("version", ARTIFACT_VERSION))
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported calibration artefact version {version}; "
                f"expected {ARTIFACT_VERSION}"
            )
        entries = [ClassCalibration.from_dict(item) for item in payload["classes"]]
        return cls(
            fold=int(payload["fold"]),
            record_count=int(payload["record_count"]),
            ground_truth_source=payload["ground_truth_source"],
            classes={entry.label: entry for entry in entries},
            default_threshold=float(payload.get("default_threshold", DEFAULT_THRESHOLD)),
            version=version,
        )

    def save(self, path: Path) -> None:
        """Write the artefact to disk as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifact:
        """Read an artefact previously written by `save`."""
        with open(path) as handle:
            return cls.from_dict(json.load(handle))


def load_default_calibration(
    path: Path = DEFAULT_CALIBRATION_PATH,
) -> CalibrationArtifact | None:
    """Load the shipped calibration artefact, or None when it is unavailable.

    Returns None when the env flag disables calibration or the file is
    missing, so the pipeline falls back to raw probabilities with the global
    threshold rather than failing.
    """
    if os.environ.get(CALIBRATION_ENV_FLAG, "1") == "0":
        logger.info("Calibration disabled via %s=0", CALIBRATION_ENV_FLAG)
        return None
    if not path.exists():
        logger.warning("Calibration artefact not found at %s — using raw outputs", path)
        return None
    try:
        return CalibrationArtifact.load(path)
    except (ValueError, KeyError, json.JSONDecodeError):
        logger.exception("Calibration artefact at %s is unreadable — using raw outputs", path)
        return None
