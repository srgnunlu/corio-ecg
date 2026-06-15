# Phase 3 Experiment: Segment-Aware v1

**Date:** 2026-06-15
**Status:** Research candidate passed; production remains experimental-only
**Factor:** Independent inference on sequential paper-ECG segments
**Version:** `segment-aware-experiment-v1`

## Scope

The experiment compares the existing `segment_ensemble` mean aggregation with
the tiled baseline using the frozen 70-image PMcardio diagnosis-drift artifact.
It does not rerun ECGFounder or change the default diagnosis path.

The locked research contract requires no regression in supported scan and
phone-photo categories and at least `+0.01` mean cosine gain in both bent and
crumpled target categories. Screens remain unclassified.

## Result

- Research candidate decision: passed
- Production status: `experimental-only`
- Failed checks: none
- Source diagnosis-drift evaluation: `70/70` successful
- Source drift artifact SHA-256:
  `8c8a6d884a54001dcce0b50784e174356f0a44cbf27a77911d3e56bcee55e355`

| Category | Role | Cosine delta | Absolute-difference delta | Agreement delta |
|---|---|---:|---:|---:|
| Bent | Target | `+0.0336` | `-0.0086` | `+0.0107` |
| Crumpled | Target | `+0.0145` | `-0.0061` | `+0.0080` |
| Doogee | Supported | `+0.0294` | `-0.0148` | `+0.0107` |
| iPhone | Supported | `+0.0261` | `-0.0095` | `+0.0107` |
| Samsung | Supported | `+0.0288` | `-0.0122` | `+0.0087` |
| Scans | Supported | `+0.0330` | `-0.0104` | `+0.0153` |

## Decision

Keep segment-aware mean aggregation as a research candidate. It consistently
reduces diagnosis-output drift relative to fake 10-second tiling, including on
the two target categories.

Do not promote it to production or the UI. This evidence measures
matched-reference diagnosis consistency, not clinical accuracy or waveform
digitization fidelity, and sparse segment inputs remain outside ECGFounder's
expected input distribution.

No model rerun occurred, so this decision adds no new runtime, timeout, or
memory measurement beyond the frozen source artifact.

## Artifacts

```text
configs/segment_aware_experiment_v1.yaml
results/digitization-experiments/segment-aware-v1/segment_aware_experiment.json
results/digitization-experiments/segment-aware-v1/segment_aware_experiment.csv
```
