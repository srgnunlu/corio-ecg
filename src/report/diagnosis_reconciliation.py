# Resolve clinically contradictory diagnoses from ECGFounder's multi-label output.
#
# ECGFounder scores its 150 labels independently (sigmoid, not softmax), so it can
# surface mutually exclusive findings at once — e.g. "sinus rhythm" AND "atrial
# fibrillation", or complete RBBB AND complete LBBB. A heart has one dominant rhythm
# and cannot carry two opposite complete bundle branch blocks, so presenting both as
# active findings is wrong. This layer resolves those conflicts BEFORE the report,
# cards, and LLM narrative are built, so every downstream view stays consistent.
#
# Two mechanisms, chosen per group for safety:
#   1. Rhythm (ARBITRATED): the deterministic rhythm analysis — interpretable P-wave
#      and RR evidence — is the referee. When it confidently reads sinus or AF, the
#      contradicting ECGFounder rhythm labels are suppressed. When it is undetermined
#      nothing is suppressed: the headline is not claiming a rhythm, so there is no
#      contradiction to resolve, and a critical label (e.g. VT) stays visible. A true
#      VT has no P waves, so the analysis can never read it as sinus — the referee
#      cannot suppress a real VT on weak grounds.
#   2. Static exclusive groups (complete bundle branch block; global normal/abnormal
#      read): the higher-probability member wins, the contradicting one is dropped.
#
# Only ABOVE-THRESHOLD members are suppressed. Sub-threshold scores never surface as
# findings, so leaving them in keeps the raw research table honest.

from __future__ import annotations

from dataclasses import dataclass

from src.measurement.rhythm_analysis import RhythmAnalysis
from src.pipeline.diagnose import DiagnosisResult

# Sinus-family rhythm labels — compatible with one another (refinements of "sinus"),
# mutually exclusive with every non-sinus mechanism below.
_SINUS_FAMILY: frozenset[str] = frozenset(
    {
        "NORMAL SINUS RHYTHM",
        "SINUS RHYTHM",
        "SINUS BRADYCARDIA",
        "SINUS TACHYCARDIA",
        "MARKED SINUS BRADYCARDIA",
    }
)

# Non-sinus primary rhythm mechanisms. Each is mutually exclusive with sinus as the
# DOMINANT rhythm. Modifiers ("WITH RAPID VENTRICULAR RESPONSE", ectopy, pacing
# qualifiers) are deliberately excluded — they qualify a rhythm, they do not compete.
_NON_SINUS_RHYTHMS: frozenset[str] = frozenset(
    {
        "ATRIAL FIBRILLATION",
        "ATRIAL FLUTTER",
        "SUPRAVENTRICULAR TACHYCARDIA",
        "MULTIFOCAL ATRIAL TACHYCARDIA",
        "VENTRICULAR TACHYCARDIA",
        "WIDE QRS TACHYCARDIA",
        "JUNCTIONAL RHYTHM",
        "JUNCTIONAL BRADYCARDIA",
        "ECTOPIC ATRIAL RHYTHM",
        "IDIOVENTRICULAR RHYTHM",
    }
)

_GOOD_QUALITY: str = "good"


@dataclass(frozen=True)
class _ExclusiveGroup:
    """A set of labels at most one of which can be true at once."""

    name: str
    members: frozenset[str]


# Static groups resolved purely by probability (no interpretable arbiter exists).
# Complete RBBB and complete LBBB cannot coexist; the model's NORMAL vs ABNORMAL
# global read is a single verdict. Fascicular/incomplete blocks are intentionally
# absent — they legitimately combine (e.g. bifascicular block).
_STATIC_GROUPS: tuple[_ExclusiveGroup, ...] = (
    _ExclusiveGroup(
        "complete_bundle_branch_block",
        frozenset({"RIGHT BUNDLE BRANCH BLOCK", "LEFT BUNDLE BRANCH BLOCK"}),
    ),
    _ExclusiveGroup(
        "global_read",
        frozenset({"NORMAL ECG", "ABNORMAL ECG"}),
    ),
)


@dataclass
class Suppression:
    """One diagnosis withheld as clinically contradictory, with the reason why."""

    label: str
    probability: float
    superseded_by: str
    reason: str


@dataclass
class ReconciliationResult:
    """Reconciled diagnoses plus an audit trail of what was suppressed and why."""

    kept: list[DiagnosisResult]
    suppressed: list[Suppression]


def reconcile_diagnoses(
    diagnoses: list[DiagnosisResult],
    rhythm: RhythmAnalysis | None,
    *,
    threshold: float,
) -> ReconciliationResult:
    """Withhold mutually exclusive diagnoses so contradictions never surface.

    Args:
        diagnoses: the full ECGFounder result list (all labels, any probability).
        rhythm: deterministic rhythm analysis; the referee for the rhythm group.
        threshold: probability cutoff defining an "active" finding — only active
            members are ever suppressed.

    Returns:
        ReconciliationResult with the kept diagnoses (suppressed labels removed) and
        the suppression audit trail for explainability.
    """
    suppressed: dict[str, Suppression] = {}

    _reconcile_rhythm(diagnoses, rhythm, threshold, suppressed)
    for group in _STATIC_GROUPS:
        _reconcile_static_group(diagnoses, group, threshold, suppressed)

    kept = [d for d in diagnoses if d.label not in suppressed]
    return ReconciliationResult(kept=kept, suppressed=list(suppressed.values()))


def _reconcile_rhythm(
    diagnoses: list[DiagnosisResult],
    rhythm: RhythmAnalysis | None,
    threshold: float,
    suppressed: dict[str, Suppression],
) -> None:
    """Let the deterministic rhythm analysis veto contradicting ECGFounder rhythms.

    Only acts on a good-quality, confident sinus or AF read. Anything else leaves the
    rhythm labels untouched — we never suppress on thin evidence.
    """
    if rhythm is None or rhythm.quality != _GOOD_QUALITY:
        return

    if rhythm.rhythm_basis == "sinus":
        targets = _NON_SINUS_RHYTHMS
        basis_text = (
            f"rhythm analysis read sinus rhythm (regular RR, P waves in "
            f"{rhythm.p_wave_fraction:.0%} of {rhythm.n_beats} beats)"
        )
    elif rhythm.rhythm_basis == "atrial_fibrillation":
        targets = _SINUS_FAMILY
        basis_text = (
            f"rhythm analysis read atrial fibrillation (irregular RR, P waves "
            f"absent across {rhythm.n_beats} beats)"
        )
    else:
        return

    for diagnosis in diagnoses:
        if (
            diagnosis.label in targets
            and diagnosis.probability >= threshold
            and diagnosis.label not in suppressed
        ):
            suppressed[diagnosis.label] = Suppression(
                label=diagnosis.label,
                probability=diagnosis.probability,
                superseded_by="rhythm analysis",
                reason=f"contradicted by {basis_text}",
            )


def _reconcile_static_group(
    diagnoses: list[DiagnosisResult],
    group: _ExclusiveGroup,
    threshold: float,
    suppressed: dict[str, Suppression],
) -> None:
    """Keep only the highest-probability active member of a mutually exclusive group."""
    members = [
        d
        for d in diagnoses
        if d.label in group.members
        and d.probability >= threshold
        and d.label not in suppressed
    ]
    if len(members) < 2:
        return

    members.sort(key=lambda d: d.probability, reverse=True)
    winner = members[0]
    for loser in members[1:]:
        suppressed[loser.label] = Suppression(
            label=loser.label,
            probability=loser.probability,
            superseded_by=winner.label,
            reason=(
                f"mutually exclusive with higher-probability {winner.label} "
                f"({winner.probability:.2f})"
            ),
        )
