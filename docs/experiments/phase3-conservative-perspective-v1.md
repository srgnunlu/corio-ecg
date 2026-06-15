# Phase 3 Experiment: Conservative Perspective v1

**Date:** 2026-06-15
**Status:** Rejected
**Factor:** High-confidence quadrilateral page rectification before digitization
**Version:** `conservative-perspective-v1`

## Scope

The experiment applies one homography only when a landscape page occupies
`45%-92%` of the image, has a supported aspect ratio, and has meaningful but
bounded perspective distortion. Vendor dewarping remains disabled.

Geometry-only review selected `3/70` PMcardio images, all crumpled:

- `img_4_page_0`
- `img_14_page_0`
- `img_81_page_0`

No bent, scan, screen, Doogee, iPhone, or Samsung image was rectified.

## Result

- Candidate digitization success: `70/70`
- Supported-category regression harness: passed
- Experiment objective: failed

| Category | Median correlation delta | RMSE delta | SNR delta |
|---|---:|---:|---:|
| Bent | `-0.00067` | `+0.00037 mV` | `-0.088 dB` |
| Crumpled | `+0.00110` | `+0.00186 mV` | `-0.024 dB` |

Corrected crumpled record correlation deltas:

- Image 81: `+0.520`
- Image 14: `-0.174`
- Image 4: `-0.621`

Image 14 runtime increased by approximately `53 seconds`.

## Decision

Reject this candidate. A quadrilateral page boundary is not sufficient evidence
that global perspective rectification will improve trace extraction on a
crumpled page. Two of three corrected records regressed substantially, and
aggregate crumpled RMSE and SNR worsened.

Do not enable this preprocessing mode in the default pipeline or UI.

## Artifacts

```text
results/digitization-experiments/perspective-v1/pmcardio_reference_fidelity.json
results/digitization-experiments/perspective-v1/regression/digitization_regression.json
```
