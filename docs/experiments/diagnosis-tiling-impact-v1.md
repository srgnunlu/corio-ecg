# Experiment: Diagnostic cost of paper-ECG segment tiling (v1)

**Date:** 2026-06-16
**Status:** Complete — isolates the input-length mismatch as a primary HR killer
and a selective diagnosis killer.
**Tooling:** `scripts/tiling_impact_eval.py`
**Result file:** `results/metrics/tiling_impact_eval.json` (n=500 PTB-XL test fold)

## Motivation

ECGFounder was trained on genuine **10 s** PTB-XL recordings — every lead carries
the full 10 s of continuous beats. A paper ECG only prints **~2.5 s per lead**
(one column of a 3x4 layout; ~5 s in some layouts). The pipeline tiles that short
segment to fill the model's 5000-sample (10 s) input. Direct inspection of cached
digitized signals confirmed the tiling and that it varies per record:
- `iphone/75`: 5 s of unique data tiled 2× (q0≈q2, q0≠q1).
- `iphone/71`: 2.5 s of unique data tiled 4× (all quarters ≈ identical) — and its
  HR reads 211 vs a true 108 (doubled).

This experiment isolates the cost of tiling **alone**, with no digitization noise,
by tiling the clean PTB-XL signals and scoring against PTB-XL ground truth.

## Method

Three arms, same 500 records, same model, `apply_rate_adjustments=False`:
- `genuine` — the real 10 s signal (identical to the baseline eval).
- `tile_2_5s` — first 2.5 s tiled 4× (a 3x4 column lead).
- `tile_5s` — first 5 s tiled 2×.

Scored on the semantic SCP subset (macro AUROC / AP, micro F1, per-class F1) and
HR vs the genuine-signal HR.

## Results

Headline diagnosis metrics (semantic SCP subset):

| arm | macro AUROC | macro AP | micro F1 |
|---|---|---|---|
| genuine | 0.877 | 0.493 | 0.646 |
| tile_2_5s | 0.832 | 0.412 | **0.529** |
| tile_5s | 0.861 | 0.443 | 0.571 |

Per-class F1 — **this is the real story**:

| class | genuine | tile_2_5s | tile_5s |
|---|---|---|---|
| SINUS RHYTHM | 0.88 | 0.88 | 0.90 |
| SINUS TACHYCARDIA | 0.91 | 0.84 | 0.88 |
| NORMAL ECG | 0.75 | 0.57 | 0.64 |
| ATRIAL FIBRILLATION | 0.79 | **0.62** | 0.68 |
| PREMATURE VENTRICULAR COMPLEXES | 0.82 | **0.25** | 0.39 |
| SINUS BRADYCARDIA | 0.33 | 0.25 | 0.31 |

Heart rate vs the genuine signal:

| arm | within 5 bpm | off > 20 bpm | median err | max err |
|---|---|---|---|---|
| tile_2_5s | 55% | **32%** | 3.2 | 159 |
| tile_5s | 69% | 22% | 1.4 | 143 |

## Interpretation

1. **Tiling severely breaks heart rate** — 32% of `tile_2_5s` cases are off by
   > 20 bpm (max 159), matching the ~26% breakage seen on real digitized signals.
   A 2.5 s segment rarely ends on a beat boundary, so repeating it injects a
   spurious periodicity that the autocorrelation HR estimator locks onto
   (frequently doubling). **HR breakage is largely a tiling artifact and is
   fixable.**

2. **Tiling is a *selective* diagnosis killer, not a global one.** Macro AUROC
   barely moves (0.877 → 0.832) and single-beat morphology calls survive intact
   (SINUS RHYTHM 0.88 → 0.88, SINUS TACHYCARDIA 0.91 → 0.84). But the diagnoses
   that **require observing many beats over the full 10 s collapse**:
   - **PVC 0.82 → 0.25** — ectopic beats are rare-per-window; tiling either repeats
     one 4× (over-counts) or drops it entirely.
   - **ATRIAL FIBRILLATION 0.79 → 0.62** — AFib is irregular irregularity; tiling a
     2.5 s block manufactures artificial regularity that masks it.

3. **Therefore tiling alone does NOT explain "can't do simple diagnoses."** On
   clean signals the model survives tiling for morphology calls. The large
   diagnosis gap users see on real photos must come from **digitization fidelity**
   (morphology/registration corruption) **stacked on top of** the tiling penalty —
   which the next eval must measure directly.

## Actionable conclusions

- **HR fix (high value, contained):** stop estimating HR on tiled leads. Use the
  genuine **10 s rhythm strip** (lead II in 3x4+1R), currently wasted by being
  tiled like every other lead; or estimate HR on the single real segment before
  tiling; or beat-align tile boundaries.
- **Temporal diagnoses (PVC/AFib):** these are structurally unrecoverable from a
  tiled 2.5 s lead. The rhythm strip is the only source of true multi-beat
  temporal information on a 3x4 paper ECG and should feed these classes.
- **Next eval:** render PTB-XL → paper image → digitize → diagnose, scored against
  PTB-XL ground-truth labels, to separate the tiling penalty (measured here) from
  the digitization-fidelity penalty on real photos.
