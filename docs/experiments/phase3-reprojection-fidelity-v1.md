# Phase 3 Experiment: Image-Space Re-projection Fidelity v1

**Date:** 2026-06-16
**Status:** Rejected — overlay fidelity is blind to the calibration/assignment failure mode
**Factor:** Reference-free agreement between the reconstruction and the detected ink
**Artifacts:** `results/pmcardio-holdout/tune/reprojection/reprojection_pilot.json`

## Rationale

Every prior probe (shadow stability, geometric perturbation, Goldberger
redundancy) failed because a digitization can be internally self-consistent yet
wrong. Following the professional/SOTA survey (PhysioNet Challenge 2024, PMcardio,
the npj/Emory at-scale work), the one *fidelity* signal that does not depend on a
ground-truth reference is agreement with the **page**: re-render the reconstructed
traces and measure how well they overlay the ink the segmentation model detected.

The segmentation probability map is upstream of polyline extraction, so the
overlay is not circular — a hallucinated, mistracked, or off-grid reconstruction
would sit off the ink (low precision) or leave ink unexplained (low recall).

## Method

- The Open-ECG-Digitizer wrapper was instrumented (vendored patch) to expose the
  segmentation probability map (`signal_probability`, the exact frame the
  polylines were extracted from) and the extractor's leading-column crop offset
  (`extraction_crop_x0`), which aligns `raw_lines` to that frame.
- `src/quality/reprojection.py::reprojection_fidelity` rasterizes the extracted
  pixel-Y polylines into the ink frame and computes, with vertical tolerances:
  **precision** (fraction of reconstructed pixels on ink), **recall** (fraction
  of ink explained by the reconstruction), **f1**, and `residual = 1 - f1`.
- Ink threshold `0.25`: the signal probability map is conservative (99.9th
  percentile ≈ `0.56`), so a `0.5` cutoff leaves an artificially sparse mask that
  makes faithful traces look off-ink. At `0.25` a faithful control reaches
  precision ≈ `1.0`.
- `scripts/reproject_one.py` digitizes one image in isolation and prints the
  scores; `scripts/reprojection_pilot.py` runs the two false accepts plus the ten
  highest true-fidelity `3x4+1R`/`3x4+3R` controls from the tune split.

## Result

Sorted by recall (ascending):

| Record | Role | True corr | Precision | Recall | Residual |
|---|---|---:|---:|---:|---:|
| scans/38 | control | `0.963` | `0.995` | `0.683` | `0.190` |
| iphone/16 | control | `0.974` | `0.992` | `0.684` | `0.191` |
| scans/83 | control | `0.975` | `0.993` | `0.684` | `0.190` |
| scans/100 | control | `0.980` | `0.995` | `0.703` | `0.176` |
| doogee/74 | **false accept** | `0.894` | `0.998` | `0.728` | `0.158` |
| samsung/16 | control | `0.981` | `0.992` | `0.745` | `0.149` |
| doogee/16 | control | `0.980` | `0.991` | `0.753` | `0.144` |
| samsung/74 | control | `0.960` | `0.998` | `0.801` | `0.111` |
| iphone/26 | **false accept** | `0.515` | `0.999` | `0.813` | `0.103` |
| scans/26 | control | `0.987` | `0.998` | `0.821` | `0.099` |
| scans/74 | control | `0.964` | `0.999` | `0.833` | `0.091` |
| scans/28 | control | `0.976` | `0.998` | `0.877` | `0.066` |

Separation margins (`min(false accept) - max(control)`; positive = clean):

| Metric | False accepts | Control range | Margin |
|---|---|---|---:|
| residual (`1 - f1`) | `0.158, 0.103` | `0.066 – 0.191` | `-0.087` |
| precision | `0.998, 0.999` | `0.991 – 0.999` | `-0.001` |
| recall | `0.728, 0.813` | `0.683 – 0.877` | `-0.150` |

Every margin is negative. The false accepts sit inside the control distribution.

## Why it fails

1. **Precision saturates (`≈ 1.0`) for every record.** The signal extractor
   *follows* the probability map by construction, so the reconstructed polyline
   trivially sits on ink. Precision therefore carries almost no fidelity
   information.
2. **Recall does not track fidelity.** `iphone/26` — the lowest-fidelity record
   in the set (true correlation `0.515`) — has the second-highest recall
   (`0.813`); its traces overlay the detected ink almost perfectly. The
   genuinely faithful `scans/38` (true `0.963`) has the lowest recall (`0.683`).
   Recall variation reflects QRS-spike ink height and trace thickness, not
   reconstruction correctness.
3. **The decisive insight — calibration/assignment errors are invisible to
   self-overlay.** `iphone/26` extracts the printed trace faithfully (it sits on
   the ink) but is wrong against the reference because of a downstream
   *interpretation* error: grid-scale calibration (mm/mV, mm/s) and/or lead
   assignment. A calibration-scale error is **self-consistent by construction** —
   re-rendering the signal with the same wrong scale lands it back on the same
   ink — so no image-space overlay can detect it. This is the same root cause as
   every prior failure: the error lives in the interpretation layer, which is
   consistent with both the signal itself and the page.

## Decision

- **Reject** image-space re-projection fidelity (overlay precision/recall/f1) as
  a quality-gate feature. Do not add it to the feature contract.
- Keep the 60-group PMcardio test split locked.
- `src/quality/reprojection.py` (with tests) and the pilot stay research-only;
  not wired into diagnosis, the UI, or the gate.
- **Keep the vendored digitizer instrumentation** (`signal_probability`,
  `extraction_crop_x0` in `vendor/patches/open-ecg-digitizer.patch`). It is
  cheap, harmless, and the only way to access the segmentation frame for future
  image-anchored work.

## What this rules in

The three reference-free families are now exhausted and, crucially, they failed
for one shared, well-localized reason: the surviving false accepts fail in the
**interpretation layer (absolute grid calibration and lead assignment)**, which
is invisible to any self-consistency or self-overlay check. The honest next
directions therefore target that layer with an *absolute physical anchor* rather
than internal agreement:

1. **Calibration-pulse / grid-scale verification.** The printed 1 mV / 10 mm
   reference pulse and the grid spacing are external physical ground truth for
   `avg_pixel_per_mm`. Detect the pulse, cross-check the implied scale, and flag
   disagreement. This is the only signal that can catch the calibration-scale
   error class. `calibration_pulse_confidence` is already listed in the contract
   under `planned_unavailable_features`.
2. **Lead-assignment verification via the detected lead labels.** Cross-check the
   Lead Name U-Net's text positions against the assigned canonical positions to
   catch assignment swaps independently of waveform morphology.
3. **Larger labelled fidelity set.** Two false accepts and thirty tune ECGs
   cannot power a reliable abstention feature; a learned confidence model should
   wait for a larger independently-collected set.
