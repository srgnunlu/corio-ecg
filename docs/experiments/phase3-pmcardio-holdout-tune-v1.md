# Phase 3 PMcardio Holdout Tune Review v1

**Date:** 2026-06-15
**Status:** Tune-only research; locked test remains unopened
**Manifest:** `results/pmcardio-holdout/pmcardio_holdout_v1.json`

## Purpose

Evaluate the frozen quality gate on previously unused PMcardio ECG groups before
any locked test review. The tune split contains 30 independent ECG groups and
210 matched physical images. The separate 60-group test split was not evaluated.

## Digitization Results

- Successful extraction: `180/210` (`85.7%`)
- Median waveform correlation: `0.5264`
- Median absolute RMSE: `0.1653 mV`
- Median SNR: `0.9248 dB`
- `12x1` extraction success: `0/21`
- `3x4+1R` median correlation: `0.882`
- `3x4+3R` median correlation: `0.879`
- `6x2` median correlation: `0.297`

The previous 70-image development result did not generalize across layouts.
Layout support is therefore a first-class quality and product-scope concern.

## Frozen Gate Results

- False accepts: `7/134` target rejects (`5.22%`)
- Reject recall: `67.91%`
- False rejects: `23/76` non-reject targets (`30.26%`)
- Non-reject coverage: `45.71%`

The frozen v1 gate is not ready for production abstention or locked test review.

## Layout-Scope Experiment

A tune-only research policy accepted only `3x4+1R` and `3x4+3R` layouts.

- False accepts: `2/134` (`1.49%`)
- False-accept bootstrap upper 95% bound: `3.85%`
- Reject recall: `95.52%`
- False rejects: `43/76` (`56.58%`)
- Non-reject coverage: `18.57%`

Narrowing the layout scope improves safety but rejects too many usable records
and still misses two low-fidelity in-scope records. It must not be promoted.

## Decision

- Keep the 60-group PMcardio test split locked.
- Do not wire the quality gate into diagnosis or the UI.
- Do not promote the layout-scope policy.
- Do not blindly tighten the existing thresholds.
- Develop a new inference-time feature that measures reconstruction or
  image-to-signal disagreement, then freeze a revised gate before one locked
  test run.

## Sample-Size Interpretation

Ten or twenty independent ECGs are useful for pipeline checks, pilots, and
finding obvious failures. They are not enough for reliable performance or
clinical claims.

The 60-group locked test is useful internal engineering evidence, but even zero
false accepts would not establish a false-accept rate below `2%`. The
rule-of-three approximation requires roughly 150 independent relevant cases
with zero failures for an upper 95% bound near `2%`. External and clinical
claims require substantially larger, independently collected datasets.
