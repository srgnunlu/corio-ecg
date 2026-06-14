# Phase 3 Quality-Gate v1 Development Baseline

**Frozen:** 2026-06-14
**Evaluation stage:** Development only
**Gate config:** `configs/quality_gate_v1.yaml`
**Feature contract:** `configs/quality_feature_contract_v1.yaml`
**Config SHA-256:** `5009b2a1bbae1702e2d6800d777c1daed60f8de62fb14ec26b9fbd6b48ea5fe5`
**Source report:** `results/pmcardio-reference/pmcardio_reference_fidelity.json`
**Source SHA-256:** `2442725bb74149bd2641ec306fb72488fcfe42199860472c99215d99e368c713`

## Scope

This baseline freezes the first interpretable quality gate on the balanced
70-image PMcardio matched-reference set. Thresholds were developed and evaluated
on the same records, so this artifact is not holdout or external evidence.

The active gate consumes only six inference-time features documented in the
versioned feature contract. Planned blur, glare, occlusion, calibration-pulse,
reconstruction-disagreement, and diagnosis-disagreement features are not yet
available and are not used by the gate.

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
| Non-reject coverage | 61.4% |
| Independent ECG groups | 10 |

ECG-group bootstrap 95% intervals:

- Reject recall: `64.0%-100.0%`
- False-reject rate: `6.7%-30.2%`
- Non-reject coverage: `54.3%-71.4%`

The zero observed false-accept rate must not be interpreted as a demonstrated
zero-risk rate because this development set contains only 10 independent ECGs.

## Confusion Matrix

| Fidelity target | Gate accept | Gate warn | Gate reject |
|---|---:|---:|---:|
| Accept | 9 | 22 | 4 |
| Warn | 2 | 6 | 5 |
| Reject | 0 | 4 | 18 |

## Missed Rejects

| ECG ID | Image ID | Category | Gate outcome |
|---|---:|---|---|
| `LPAE_09754_hr` | 43 | Crumpled | Warn |
| `LPAE_14680_hr` | 92 | Screen | Warn |
| `LPAE_17226_hr` | 14 | iPhone | Warn |
| `LPAE_17226_hr` | 14 | Scan | Warn |

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
