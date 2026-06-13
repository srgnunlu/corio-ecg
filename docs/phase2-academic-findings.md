# Phase 2 Round-Trip Findings

## Validity Notice

The historical n=500 Phase 2 results are invalid for scientific interpretation.
They were produced before the local Net1D implementation was proven equivalent
to ECGFounder's official implementation. The old model used BatchNorm during
inference and differed in first-block activation and residual channel padding.

On 2026-06-13, the local model was corrected and produced exactly equal logits
to the official implementation on the same checkpoint and input
(`max_abs_diff = 0.0`). Only results produced after that correction should be
used.

## Corrected Diagnostic Baseline

Full PTB-XL test fold, raw model probabilities, threshold 0.5:

| Metric | Result |
|---|---:|
| Test records processed | 2,203 / 2,203 |
| Records matched to official targets | 2,198 |
| Eligible official classes (minimum 5 positives) | 31 / 150 |
| Official macro AUROC | 0.8679 |
| Official macro average precision | 0.4454 |
| Official micro F1 | 0.5885 |
| Official macro F1 | 0.3673 |

This establishes that the diagnostic model integration is functioning. It does
not establish clinical validity.

## Audited Round-Trip Smoke Benchmark

The available corrected round-trip set contains only 50 records. All 150
digitized outputs were regenerated with synchronized canonical segment
expansion and paired pipeline-version metadata. With a minimum support of five
positives, only three classes are eligible, so diagnostic deltas are
directional rather than statistically robust.

| Scenario | Cosine similarity | Agreement | Mean Pearson | Official micro F1 delta |
|---|---:|---:|---:|---:|
| Clean | 0.5453 | 94.80% | 0.2175 | -0.2032 |
| Moderate | 0.5394 | 94.67% | 0.2061 | -0.1922 |
| Hard | 0.5492 | 94.56% | 0.1850 | -0.1922 |

High overall agreement is dominated by the many negative classes and must not
be interpreted as preserved diagnostic accuracy. The F1 drop and low signal
correlation make digitization fidelity the current primary technical risk.

Physical consistency now degrades as expected with image difficulty:

| Scenario | Mean Einthoven consistency | Outputs with 12 active leads |
|---|---:|---:|
| Clean | 0.9893 | 50 / 50 |
| Moderate | 0.9473 | 49 / 50 |
| Hard | 0.8744 | 48 / 50 |

The similar diagnostic drift across all three scenarios suggests that the
dominant remaining error is paper-layout reconstruction, not image noise alone.
A standard 3x4 sheet contains sequential 2.5-second lead segments, while
ECGFounder expects a simultaneous 10-second 12-lead tensor.

## Current Interpretation

- ECGFounder inference now matches the upstream implementation.
- The official PTB-XL target vectors are now the primary diagnostic benchmark.
- The semantic SCP-code mapping remains useful as a separately named subset
  metric.
- The next scientifically meaningful milestones are a segment-aware
  reconstruction strategy, a larger corrected round-trip benchmark, and
  real-phone-photo validation.
- No clinical-performance or production-readiness claim is supported.

## Source Artifacts

- `results/metrics/ptbxl_baseline_summary.json`
- `results/metrics/roundtrip_comparison.json`
- `docs/evaluation-methodology.md`
