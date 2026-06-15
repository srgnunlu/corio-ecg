"""Evaluate segment-aware diagnosis consistency under a research-only contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENT_AWARE_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "configs" / "segment_aware_experiment_v1.yaml"
)
METRICS = (
    "mean_cosine_similarity",
    "mean_abs_probability_difference",
    "mean_agreement_rate",
)


@dataclass(frozen=True)
class SegmentAwareTolerances:
    """Locked research-candidate tolerances."""

    mean_cosine_similarity_drop: float
    mean_abs_probability_difference_increase: float
    mean_agreement_rate_drop: float
    target_mean_cosine_similarity_gain: float

    def validate(self) -> None:
        """Reject negative tolerances."""
        if any(value < 0.0 for value in asdict(self).values()):
            raise ValueError("segment-aware experiment tolerances must be non-negative")


@dataclass(frozen=True)
class SegmentAwareExperimentConfig:
    """Versioned methods, category roles, and research tolerances."""

    version: str
    baseline_method: str
    candidate_method: str
    supported_categories: tuple[str, ...]
    target_categories: tuple[str, ...]
    tolerances: SegmentAwareTolerances

    def validate(self) -> None:
        """Validate methods, roles, and tolerances."""
        if not all((self.version.strip(), self.baseline_method, self.candidate_method)):
            raise ValueError("segment-aware experiment version and methods must not be empty")
        if self.baseline_method == self.candidate_method:
            raise ValueError("baseline and candidate methods must differ")
        if not self.supported_categories or not self.target_categories:
            raise ValueError("supported and target categories are required")
        overlap = set(self.supported_categories) & set(self.target_categories)
        if overlap:
            raise ValueError(f"category roles overlap: {sorted(overlap)}")
        self.tolerances.validate()

    def with_categories(
        self,
        *,
        supported_categories: tuple[str, ...],
        target_categories: tuple[str, ...],
    ) -> SegmentAwareExperimentConfig:
        """Return a validated copy with focused category roles."""
        config = replace(
            self,
            supported_categories=supported_categories,
            target_categories=target_categories,
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""
        return asdict(self)


def load_segment_aware_experiment_config(
    path: Path = DEFAULT_SEGMENT_AWARE_EXPERIMENT_CONFIG,
) -> SegmentAwareExperimentConfig:
    """Load and validate the segment-aware research contract."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("segment-aware experiment config must contain a mapping")
    try:
        config = SegmentAwareExperimentConfig(
            version=str(payload["version"]),
            baseline_method=str(payload["baseline_method"]),
            candidate_method=str(payload["candidate_method"]),
            supported_categories=tuple(payload["supported_categories"]),
            target_categories=tuple(payload["target_categories"]),
            tolerances=SegmentAwareTolerances(**payload["tolerances"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid segment-aware experiment config structure: {exc}") from exc
    config.validate()
    return config


def _category_comparison(
    aggregate: dict[str, Any],
    config: SegmentAwareExperimentConfig,
    role: str,
) -> dict[str, Any]:
    methods = aggregate["methods"]
    baseline = methods[config.baseline_method]
    candidate = methods[config.candidate_method]
    deltas = {
        metric: float(candidate[metric]) - float(baseline[metric]) for metric in METRICS
    }
    return {
        "role": role,
        "source_counts": {
            key: aggregate.get(key) for key in ("total", "successful", "failed")
        },
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
    }


def _failed_checks(
    categories: dict[str, dict[str, Any]],
    config: SegmentAwareExperimentConfig,
) -> list[dict[str, Any]]:
    tolerance = config.tolerances
    rules = {
        "mean_cosine_similarity": -tolerance.mean_cosine_similarity_drop,
        "mean_abs_probability_difference": tolerance.mean_abs_probability_difference_increase,
        "mean_agreement_rate": -tolerance.mean_agreement_rate_drop,
    }
    failed: list[dict[str, Any]] = []
    for category in (*config.supported_categories, *config.target_categories):
        result = categories.get(category)
        role = "supported" if category in config.supported_categories else "target"
        if result is None:
            failed.append(
                {
                    "category": category,
                    "role": role,
                    "metric": "category",
                    "reason": "missing",
                }
            )
            continue
        for metric, limit in rules.items():
            observed = result["deltas"][metric]
            is_abs_difference = metric == "mean_abs_probability_difference"
            if (is_abs_difference and observed > limit + 1e-12) or (
                not is_abs_difference and observed < limit - 1e-12
            ):
                failed.append(
                    {
                        "category": category,
                        "role": role,
                        "metric": metric,
                        "observed_change": observed,
                        "allowed_change": limit,
                    }
                )
        target_gain = result["deltas"]["mean_cosine_similarity"]
        if role == "target" and target_gain < tolerance.target_mean_cosine_similarity_gain - 1e-12:
            failed.append(
                {
                    "category": category,
                    "role": role,
                    "metric": "target_mean_cosine_similarity_gain",
                    "observed_change": target_gain,
                    "required_change": tolerance.target_mean_cosine_similarity_gain,
                }
            )
    return failed


def evaluate_segment_aware_experiment(
    drift_report: dict[str, Any],
    *,
    config: SegmentAwareExperimentConfig | None = None,
) -> dict[str, Any]:
    """Evaluate an existing diagnosis-drift report without rerunning the model."""
    resolved = config or load_segment_aware_experiment_config()
    source_categories = drift_report["aggregate"]["categories"]
    categories: dict[str, dict[str, Any]] = {}
    for category, aggregate in source_categories.items():
        role = (
            "supported"
            if category in resolved.supported_categories
            else "target"
            if category in resolved.target_categories
            else "unclassified"
        )
        categories[category] = _category_comparison(aggregate, resolved, role)
    failed = _failed_checks(categories, resolved)
    return {
        "config": resolved.to_dict(),
        "research_candidate": {"passed": not failed, "failed_checks": failed},
        "production_status": "experimental-only",
        "limitations": (
            "This decision measures matched-reference diagnosis consistency, not clinical "
            "accuracy or digitization fidelity. Sparse segment inputs remain outside the "
            "diagnosis model's expected input distribution."
        ),
        "categories": categories,
    }


__all__ = [
    "DEFAULT_SEGMENT_AWARE_EXPERIMENT_CONFIG",
    "evaluate_segment_aware_experiment",
    "load_segment_aware_experiment_config",
]
