# Phase 3 Experiment: Goldberger Limb-Lead Consistency v1

**Date:** 2026-06-16
**Status:** Rejected — internal limb-lead consistency does not separate the failures
**Factor:** Absolute physiological plausibility (Einthoven + Goldberger redundancy)
**Artifacts:** `results/pmcardio-holdout/tune/physiological-consistency/goldberger_pilot.json`

## Rationale

The shadow and perturbation pilots were rejected because both probed
*stability*, not *fidelity*: a wrong-but-stable digitization slipped through.
This pilot changes the angle to *absolute physiological plausibility*. A
faithful 12-lead reconstruction must satisfy the four Einthoven/Goldberger
limb-lead derivation identities; a distorted or mis-assigned one should violate
them no matter how reproducible it is.

It extends the single Einthoven check already in the gate (`II = I + III`) to the
full limb-lead redundancy set:

- `II  = I + III`        (Einthoven)
- `aVR = -(I + II) / 2`  (Goldberger)
- `aVL = (I - III) / 2`  (Goldberger)
- `aVF = (II + III) / 2` (Goldberger)

## Method

- `src/utils/signal_clean.py::goldberger_consistency` scores a digitized signal
  against the four identities. Each rule's measured lead is matched to its
  derivation with a **best-lag normalized cross-correlation** (phase-tolerant),
  and the residual is `1 - correlation` clamped to `[0, 1]` (higher = worse).
  Aggregates: `mean_residual` and `worst_residual` over the four rules.
- Phase tolerance is required because on a 3x4 paper ECG the limb leads come
  from different columns (different 2.5 s time windows). After column tiling the
  augmented leads are phase-shifted from `I/II/III` by an arbitrary fraction of
  a cardiac cycle, so a zero-lag residual would punish faithful records. Two lag
  variants are reported: `lag_tolerant` (±1.3 s) and `zero_lag`.
- `scripts/goldberger_pilot.py` digitizes each record once (isolated no-retry
  subprocess, ~18 GB peak) and scores the calibrated mV signal. Records: the two
  layout-in-scope false accepts plus the ten highest true-fidelity `3x4+1R` /
  `3x4+3R` controls from the tune split.

## Result

Lag-tolerant residuals, sorted ascending (lower = more internally consistent):

| Record | Role | True corr | Mean resid | Worst resid | II | aVR | aVL | aVF |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iphone/26 | **false accept** | `0.515` | `0.135` | `0.201` | `0.02` | `0.20` | `0.15` | `0.16` |
| scans/38 | control | `0.963` | `0.146` | `0.220` | `0.03` | `0.22` | `0.20` | `0.14` |
| scans/83 | control | `0.975` | `0.249` | `0.362` | `0.00` | `0.30` | `0.33` | `0.36` |
| iphone/16 | control | `0.974` | `0.353` | `0.501` | `0.06` | `0.47` | `0.38` | `0.50` |
| scans/28 | control | `0.976` | `0.388` | `0.660` | `0.04` | `0.66` | `0.48` | `0.37` |
| samsung/16 | control | `0.981` | `0.389` | `0.589` | `0.02` | `0.49` | `0.45` | `0.59` |
| doogee/16 | control | `0.980` | `0.399` | `0.608` | `0.01` | `0.52` | `0.46` | `0.61` |
| scans/74 | control | `0.964` | `0.421` | `0.660` | `0.01` | `0.44` | `0.66` | `0.57` |
| samsung/74 | control | `0.960` | `0.430` | `0.639` | `0.06` | `0.47` | `0.64` | `0.56` |
| doogee/74 | **false accept** | `0.894` | `0.497` | `0.797` | `0.01` | `0.42` | `0.80` | `0.77` |
| scans/26 | control | `0.987` | `0.552` | `0.809` | `0.33` | `0.81` | `0.73` | `0.34` |
| scans/100 | control | `0.980` | `0.663` | `0.826` | `0.76` | `0.83` | `0.31` | `0.75` |

Separation margins (`min(false accept) - max(control)`; positive = clean
separation):

| Variant | Metric | False accepts | Control range | Margin |
|---|---|---|---|---:|
| lag_tolerant | mean_residual | `0.135, 0.497` | `0.146 – 0.663` | `-0.528` |
| lag_tolerant | worst_residual | `0.201, 0.797` | `0.220 – 0.826` | `-0.625` |
| zero_lag | mean_residual | `0.737, 0.743` | `0.462 – 0.986` | `-0.248` |
| zero_lag | worst_residual | `1.000, 1.000` | `0.714 – 1.000` | `0.000` |

Every margin is non-positive. The false accepts are interleaved among the
controls, not above them.

## Why it fails

Two independent reasons, both fatal:

1. **Internal consistency is not fidelity.** `iphone/26` is the *lowest*-fidelity
   record in the set (true correlation `0.515`) yet has the *lowest* Goldberger
   residual of all twelve (`0.135`). Its limb leads are mutually self-consistent
   while the reconstruction as a whole is wrong — the wrong-but-stable failure
   mode again, now in a new disguise. The redundancy identities only constrain
   the six limb leads; they say nothing about V1–V6 or about absolute
   correctness, so a record with plausible limb leads and poor chest leads sails
   through.
2. **The augmented identities are temporally invalid on column-tiled paper.**
   `aVR/aVL/aVF` live in a different paper column (a different 2.5 s window, i.e.
   different beats) than `I/II/III`. Even with ±1.3 s phase search, their
   residuals stay high (`0.3–0.8`) on genuinely faithful controls (true
   correlation `> 0.96`), because the leads being compared are not simultaneous
   recordings of the same beat. The clean rule is `II = I + III`, where all
   three leads share one column — and that single rule is already in the gate.

Note `scans/26` and `scans/100` (true correlation `> 0.98`) post the *highest*
residuals, including on the Einthoven `II` rule, confirming the metric tracks
acquisition geometry and tiling artefacts rather than reconstruction fidelity.

## Decision

- **Reject** Goldberger limb-lead consistency as a quality-gate feature. Do not
  add `mean_goldberger_residual` / `worst_goldberger_residual` to the feature
  contract.
- Keep the 60-group PMcardio test split locked.
- `goldberger_consistency` stays in `src/utils/signal_clean.py` as a research /
  diagnostic tool (with tests); it is **not** wired into diagnosis, the UI, or
  the gate. The existing single Einthoven `II` check is unchanged.

## Alternative strategies to explore next

1. **Reference-anchored fidelity, not internal redundancy.** Every probe that
   uses only the digitized signal (stability, redundancy) has now failed because
   a wrong reconstruction can be internally clean. The remaining honest signal
   of fidelity is agreement with the *image* (re-render the digitized signal as a
   trace and measure overlap with the extracted pixels) or with an independent
   second digitizer, not internal self-consistency.
2. **Layout-scope plus mandatory review** as the conservative near-term product
   policy, accepting low coverage, until a larger labelled fidelity set exists.
3. **Larger labelled fidelity set.** A reliable abstention feature needs more
   than 30 tune ECGs; with only two false accepts, any single-feature pilot is
   underpowered. Defer a learned confidence model until such a set is collected.
