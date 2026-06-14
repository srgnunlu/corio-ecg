# Phase 3 Quality-Gate v1 Development Baseline

**Frozen:** 2026-06-14
**Evaluation stage:** Development only
**Gate config:** `configs/quality_gate_v1.yaml`
**Config SHA-256:** `5009b2a1bbae1702e2d6800d777c1daed60f8de62fb14ec26b9fbd6b48ea5fe5`
**Source report:** `results/pmcardio-reference/pmcardio_reference_fidelity.json`
**Source SHA-256:** `2442725bb74149bd2641ec306fb72488fcfe42199860472c99215d99e368c713`

## Scope

This baseline freezes the first interpretable quality gate on the balanced
70-image PMcardio matched-reference set. Thresholds were developed and evaluated
on the same records, so this artifact is not holdout or external evidence.

## Frozen Aggregate

| Metric | Value |
|---|---:|
| Records | 70 |
| Fidelity targets: accept / warn / reject | 35 / 13 / 22 |
| Gate outcomes: accept / warn / reject | 11 / 32 / 27 |
| False accepts | 0 |
| False rejects | 9 |
| Missed rejects | 4 |
| Reject recall | 81.8% |

## Category Results

| Category | Records | False accepts | False rejects | Missed rejects | Reject recall |
|---|---:|---:|---:|---:|---:|
| Bent | 10 | 0 | 1 | 0 | 100.0% |
| Crumpled | 10 | 0 | 2 | 1 | 87.5% |
| Doogee | 10 | 0 | 1 | 0 | N/A |
| iPhone | 10 | 0 | 1 | 1 | 50.0% |
| Samsung | 10 | 0 | 1 | 0 | N/A |
| Scans | 10 | 0 | 0 | 1 | 50.0% |
| Screens | 10 | 0 | 3 | 1 | 50.0% |

## Regeneration

```bash
source .venv/bin/activate
python scripts/evaluate_quality_gate.py \
  --evaluation-stage development \
  --expected-input-sha256 2442725bb74149bd2641ec306fb72488fcfe42199860472c99215d99e368c713 \
  --allow-overwrite
```

The `--allow-overwrite` flag must only be used for an intentional regeneration
from the exact frozen source. Any changed source requires a new baseline version.
