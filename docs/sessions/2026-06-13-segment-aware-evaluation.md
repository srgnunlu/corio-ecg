# Segment-Aware Diagnosis Evaluation

## Objective

Test whether independent inference on the four sequential columns of a standard
3x4 paper ECG reduces the diagnosis drift caused by presenting tiled printed
segments as a simultaneous 12-lead recording.

## Implementation

- Added paper-column model input construction using the existing 3x4 and 6x2
  lead layout maps.
- Added mean and maximum probability aggregation.
- Added an experimental `segment_ensemble` result beside the existing tiled
  round-trip result.
- Added CLI controls:
  - `--paper-layout {3x4,6x2}`
  - `--segment-aggregation {mean,max}`
- Kept production and Gradio diagnosis behavior unchanged.

## Audited 50-Record Result

Mean aggregation was selected after a 5-record smoke comparison. Maximum
aggregation reduced agreement and produced larger probability differences.

| Scenario | Tiled cosine | Segment cosine | Tiled agreement | Segment agreement |
|---|---:|---:|---:|---:|
| Clean | 0.5453 | 0.7657 | 94.80% | 97.56% |
| Moderate | 0.5394 | 0.7689 | 94.67% | 97.67% |
| Hard | 0.5492 | 0.7532 | 94.56% | 97.37% |

Official matched-baseline metric deltas:

| Scenario | Tiled micro F1 delta | Segment micro F1 delta | Tiled macro AUROC delta | Segment macro AUROC delta |
|---|---:|---:|---:|---:|
| Clean | -0.2032 | -0.1104 | -0.0967 | +0.0086 |
| Moderate | -0.1922 | -0.1299 | -0.1015 | +0.0152 |
| Hard | -0.1922 | -0.1201 | -0.1029 | +0.0073 |

Current result:
`results/metrics/roundtrip_segment_ensemble_comparison.json`

## Interpretation

The result supports the paper-layout timing hypothesis: independent inference
on sparse paper columns substantially reduces diagnosis drift relative to the
current tiled representation.

The 50-record benchmark has limited class support and is not a robust accuracy
estimate. Sparse column inputs are also outside ECGFounder's expected input
distribution. The method remains an evaluation experiment and must not replace
production diagnosis until it is tested on a larger matched benchmark.

## Verification

- Segment-aware unit tests: 6 passed.
- Focused evaluation tests: 11 passed.
- Focused Ruff checks: passed.
- Existing round-trip result file was not overwritten.
