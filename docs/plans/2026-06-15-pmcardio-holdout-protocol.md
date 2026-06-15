# PMcardio Internal Holdout Protocol

**Pre-registered:** 2026-06-15
**Manifest:** `results/pmcardio-holdout/pmcardio_holdout_v1.json`
**Evidence level:** Internal engineering holdout, not external or clinical evidence

## Scope

The PMcardio physical-photo collection contains 100 independent ECG groups.
Each ECG has seven matched physical capture variants across six paper layouts.
The 10 ECG groups used for development and threshold design are excluded.

The remaining 90 previously unseen ECG groups are locked before evaluation:

- `30` tune groups for future threshold or feature development;
- `60` locked test groups for Phase 3 engineering decisions;
- `630` matched images in total.

All variants of one ECG remain in one split. The locked test split must not be
used for threshold tuning, feature selection, or experiment selection.

## What This Sample Can Support

The 60-group locked test split is sufficient to expose large regressions,
compare supported categories, and determine whether the current quality gate is
ready for a larger external study.

It cannot establish the Phase 3 false-accept target below `2%`. With zero
observed failures, the rule-of-three approximation requires roughly 150
independent relevant cases to place the upper 95% bound near `2%`. Correlated
photo variants do not replace independent ECG groups.

Clinical or product claims require a larger external dataset with different
sites, devices, printers, populations, and independently adjudicated outcomes.

## Execution Order

1. Commit the pre-registration manifest before evaluation.
2. Download only manifest-listed images.
3. Evaluate or tune only on the `tune` split.
4. Freeze any revised gate or candidate.
5. Run the locked quality-gate review once on the `test` split.
6. Report ECG-group bootstrap intervals and category/layout results.
