# Phase 3 Experiment: Conservative Shadow v1

**Date:** 2026-06-15
**Status:** Rejected by locked promotion gate
**Factor:** Broad illumination normalization after safe image downscaling
**Version:** `conservative-shadow-v1`

## Scope

The experiment estimates a smooth illumination map and applies bounded gain
normalization only when a bright ECG page has a broad luminance gradient.
Perspective correction and vendor dewarping remain disabled.

Geometry-only review selected most phone, screen, bent, and crumpled images,
but selected no scan images.

## Result

- Candidate digitization success: `70/70`
- Supported-category fidelity tolerances: passed
- Locked promotion gate: failed on Doogee runtime

| Category | Correlation delta | RMSE delta | SNR delta | Runtime delta |
|---|---:|---:|---:|---:|
| Bent | `+0.1749` | `-0.0085 mV` | `+0.359 dB` | `+2.5%` |
| Crumpled | `+0.0960` | `+0.0009 mV` | `+0.696 dB` | `+14.5%` |
| Doogee | `-0.0000` | `-0.0007 mV` | `-0.050 dB` | `+33.2%` |
| iPhone | `+0.0030` | `-0.0103 mV` | `+0.106 dB` | `-13.1%` |
| Samsung | `+0.0017` | `+0.0005 mV` | `+0.186 dB` | `-8.9%` |
| Scans | `-0.0003` | `+0.0001 mV` | `-0.015 dB` | `-3.0%` |

The normalization function itself takes approximately `0.06 seconds` on a
downscaled Doogee image. The observed Doogee wall-time increase occurred
without extra orientation retries, but still exceeds the locked `25%`
category tolerance and cannot be waived after evaluation.

Screens are not a supported promotion category and regressed materially:
correlation `-0.0429`, RMSE `+0.0203 mV`, and SNR `-0.738 dB`.

## Decision

Reject promotion under the current locked contract. Keep the mode experimental
because target-category fidelity improved substantially, but do not enable it
in the default pipeline or UI.

A future revision must narrow the selected capture conditions and use a
pre-registered, more stable runtime measurement before reevaluation.

## Artifacts

```text
results/digitization-experiments/shadow-v1/pmcardio_reference_fidelity.json
results/digitization-experiments/shadow-v1/regression/digitization_regression.json
```
