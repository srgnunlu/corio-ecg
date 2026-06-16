# Phase 3 Experiment: Grid-Preserving Perturbation Pilot v1

**Date:** 2026-06-16
**Status:** Rejected — perturbation disagreement does not separate the failures
**Factor:** Reconstruction stability under bounded, grid-preserving geometry
**Artifacts:** `results/pmcardio-holdout/tune/reconstruction-disagreement/perturbation_pilot.json`

## Rationale

The shadow perturbation pilot was rejected because the two layout-in-scope
false accepts (doogee img_74, iphone img_26) stayed extremely stable under
illumination normalization. This pilot tests a stronger probe: a bounded,
clinically irrelevant geometric perturbation that preserves the ECG grid.

If a low-fidelity extraction is fragile, a small scale or shift of the source
photo should move its reconstruction more than it moves a faithful one.

## Method

- `src/evaluation/perturbation_harness.py` applies bounded, grid-preserving
  perturbations: integer-pixel translation (edge-replicated, interpolation
  free) and a single bilinear center scale. Bounds: scale `[0.95, 1.05]`,
  translation `<= 20 px`.
- Probe set: `translate (+12, -8)`, `scale 1.03`, `scale 0.97 + translate (-10, +10)`.
- `scripts/perturbation_pilot.py` digitizes an unperturbed reference arm and
  each perturbed arm, then scores them with `evaluate_reconstruction_pair`.
  The worst (least stable) probe per record is reported.
- Records: the two false accepts plus the ten highest true-fidelity in-scope
  (`3x4+1R` / `3x4+3R`) controls from the tune split.
- Each digitization runs in an isolated subprocess with retries disabled
  (`scripts/digitize_one.py --no-retries`). The dewarping retry balloons the
  CPU working set past 100 GB and was OOM-killed; a single deterministic pass
  keeps the footprint near 18 GB and is the correct arm definition for a
  stability probe (both arms share the setting, so the comparison stays fair).

## Result

Worst-case stability (lower = less stable) against true reference fidelity:

| Record | Role | True corr | Worst median corr | Unstable leads | Cal. RMSE mV |
|---|---|---:|---:|---:|---:|
| samsung img_74 | control | `0.960` | `0.789` | `7` | `0.196` |
| iphone img_26 | false accept | `0.515` | `0.797` | `6` | `0.152` |
| doogee img_74 | false accept | `0.894` | `0.847` | `4` | `0.171` |
| samsung img_16 | control | `0.981` | `0.922` | `5` | `0.136` |
| scans img_100 | control | `0.980` | `0.947` | `2` | `0.056` |
| scans img_74 | control | `0.964` | `0.966` | `1` | `0.085` |
| (remaining 5 scans/phone controls) | control | `>= 0.963` | `>= 0.988` | `<= 3` | `<= 0.034` |

Separation margin (lowest control stability minus highest false-accept
stability) = `0.789 - 0.847 = -0.058`. The distributions overlap, so no
threshold cleanly separates the failures.

## Why it fails

The probe measures perturbation sensitivity, which tracks capture condition and
layout geometry, not reconstruction fidelity. The single most unstable record
in the set is `samsung img_74` — a genuinely faithful extraction (true
correlation `0.960`). Meanwhile `doogee img_74`'s true fidelity is only
moderately low (`0.894`), so it does not stand out. The false accepts are
stable reconstructions of a poor reading: stability is not fidelity, exactly
the guardrail noted in `phase3-reconstruction-disagreement-v1.md`.

## Decision

- Reject grid-preserving geometric perturbation as a quality-gate feature.
- Keep the 60-group PMcardio test split locked.
- The metric (`reconstruction_disagreement.py`) and harness stay as research
  tools; do not wire them into diagnosis, the UI, or the gate.

## Alternative strategies to explore next

1. **Absolute physiological plausibility, not stability.** Extend the
   Einthoven check to the full limb-lead redundancy set (Goldberger:
   `aVR ~= -(I + II) / 2`, `aVL ~= (I - III) / 2`, `aVF ~= (II + III) / 2`).
   A reconstruction that violates several lead-derivation identities is
   internally inconsistent regardless of how stable it is. This attacks the
   failure mode directly (a wrong but stable reading) instead of probing
   sensitivity.
2. **Layout-scope plus mandatory review** as the conservative near-term
   product policy, accepting low coverage, until a larger labelled fidelity
   set exists.
3. **Larger labelled fidelity set.** A reliable abstention feature needs more
   than 30 tune ECGs; the current overlap may also reflect sample noise.
   Defer a learned confidence model until such a set is collected.
