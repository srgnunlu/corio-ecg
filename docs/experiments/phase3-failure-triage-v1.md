# Phase 3 Experiment: Failure Triage (v1)

**Date:** 2026-06-16
**Status:** Complete — decides Track A vs Track B priority
**Plan:** `docs/plans/phase3-calibration-anchor-plan.md` §3 (Phase 0)
**Method:** No re-digitization. Reused the cached holdout-tune signals
(`data/reference/pmcardio-holdout/digitized/` and `digitized-calibrated/`) plus
the matched printed references (`leads.npz`). Tooling:
`scripts/failure_triage.py`.

## Question

Are the two surviving tune false accepts (`iphone/26`, `doogee/74`) wrong because
of (a) a global **amplitude-scale** error — the Track A calibration anchor — or
(b) a **lead permutation / assignment** error — the Track B label anchor — or
(c) neither (morphology/extraction)?

Two diagnostics per record:
- **Permutation:** the 12×12 best-shifted-correlation matrix between z-scored
  digitized and reference leads. An off-diagonal argmax that beats the diagonal
  by > 0.10 flags a misassigned lead.
- **Scale:** per-lead calibrated RMS gain ratios (digitized / reference). A tight
  cluster (low CV) around a constant `k ≠ 1` is a global-scale error; scatter is
  not.

Two clean controls were included, one of them the **same ECG** as `iphone/26`
captured by a flatbed scan (`scans/26`, ref `LPAE_20971_hr`) — a natural
controlled comparison.

## Results

| record | diag corr (median / min) | permuted leads | gain median | gain CV | dev from 1.0 |
|---|---|---|---|---|---|
| **iphone/26** (LPAE_20971) | 0.515 / 0.024 | **6** | 0.948 | 0.040 | 0.052 |
| **doogee/74** (LPAE_12629) | 0.894 / 0.527 | 0 | 0.828 | 0.222 | 0.172 |
| scans/26 — control (LPAE_20971) | 0.987 / 0.971 | 0 | 0.946 | 0.019 | 0.054 |
| doogee/16 — control (LPAE_19550) | 0.980 / 0.199 | 0 | 0.938 | 0.306 | 0.062 |

### iphone/26 — horizontal/temporal **registration** error, not scale or label

- **Scale is fine.** Gain ratios are tight (CV 0.040) at median 0.948 — *identical*
  to the clean scan of the **same ECG** (0.946, CV 0.019). A 1 mV pulse-amplitude
  anchor would read ≈ 0.95 mV for both the false accept and the clean control, so
  **Track A cannot separate `iphone/26`.**
- **Labels are fine.** The digitizer detected V1–V6 (and II, III, aVL, aVF) with
  `layout_cost` 0.22 (low = confident assignment) and `einthoven_score` 0.957.
  The per-lead energies are *identical* to the clean scan (V1 1.776 vs 1.802,
  V3 1.284 vs 1.315). The right content, with the right amplitude, is in each
  precordial slot — so a label-position check (Track B) would **pass** and miss
  the error.
- **The defect is horizontal/temporal registration.** The digitized precordial
  morphology is intact — the internal V1≈V2≈V3 and V4≈V5≈V6 correlation structure
  is *identical* to the clean scan. But each precordial cell's content sits at the
  **wrong sample offset** inside the canonical lead. Sliding the reference window
  across the full digitized lead recovers high correlation only at a large offset:

  | lead | corr at window `[:1250]` | best corr | best offset |
  |---|---|---|---|
  | V1 | −0.04 | 0.92 | 2560 samp (5.12 s) |
  | V2 | −0.04 | 0.89 | 2560 samp (5.12 s) |
  | V4 | −0.15 | 0.87 | 2580 samp (5.16 s) |

  The clean scan recovers its best correlation at offset 0 (or an exact 2.5 s
  period multiple). So `iphone/26`'s precordial cells extracted the correct lead's
  correct shape but placed it ~5 s out of position — a per-cell horizontal
  (column/time-window) registration error, almost certainly driven by perspective
  / scale distortion on the phone photo. The ±100 ms alignment in the fidelity
  metric cannot cross a ~5 s offset, which is why the diagonal correlation
  collapses to ≈ 0.

  This error is invisible to **both** planned anchors: amplitude is correct
  (Track A blind) and lead labels are correct (Track B blind).

### doogee/74 — neither scale nor assignment (marginal morphology)

- **No permutation** (every lead's argmax is on the diagonal; V5→V6 is +0.02, below
  threshold). Lead mapping is correct.
- **Not a global scale error:** gains scatter (CV 0.222) with no single `k`
  (lead I 0.455, aVF 1.218). But scatter alone is weak — control `doogee/16` has
  even higher CV (0.306) at diag 0.980.
- Ten of twelve leads are on-diagonal at 0.89–0.99; only **V4–V6 are mediocre**
  (0.53–0.56). This is a correctly-mapped, correctly-scaled digitization with
  degraded precordial morphology — a marginal near-miss, **catchable by neither
  physical anchor.**

## Decision

The triage **refutes both planned tracks** for the dominant false accept
(`iphone/26`). This is the §9 "documented rejection" outcome the plan anticipated,
but it also pinpoints a *new*, more promising anchor.

1. **Track A (calibration-pulse amplitude) — refuted.** Neither survivor has a
   global amplitude-scale error. `iphone/26`'s calibration is as good as a clean
   scan of the same ECG; `doogee/74`'s gain scatter is not single-`k` and is no
   worse than a high-fidelity control. A 1 mV pulse-deviation feature has no
   separation to exploit.
2. **Track B (lead-label assignment) — refuted for these survivors.**
   `iphone/26`'s labels, layout assignment (`layout_cost` 0.22), and per-lead
   energies are all correct/identical to the clean scan. The error is *not* which
   lead sits in which cell — it is *where in time* each cell's content was placed.
   A label-position check would pass and miss it. (Track B might still catch a
   *different* class of false accept on the locked test, but it cannot catch the
   survivors we have.)
3. **Real mechanism: per-cell horizontal/temporal registration.** `iphone/26`'s
   precordial cells hold the correct lead with the correct shape and amplitude,
   shifted ~5 s out of position. This is the detectable defect — and it is blind
   to both amplitude and label anchors.
4. **`doogee/74` is out of reach of all three** (no scale, no permutation, no gross
   misregistration — just degraded V4–V6 morphology). Treat as a marginal
   near-miss, not a gross interpretation-layer false accept.

## Recommended next direction (replaces Tracks A/B)

A **temporal-registration consistency anchor**, reference-free:

- For rhythm-strip layouts (`3x4+1R`/`3x4+3R`), lead II appears both as a column
  segment and as the full-length rhythm strip. The column segment must temporally
  align with the matching window of the rhythm strip; a per-cell horizontal
  registration error breaks that alignment **without any reference signal**.
- More generally: cells in the same printed column share one acquisition window;
  their R-wave timings must be mutually consistent. A cell shifted ~5 s
  (`iphone/26`) violates same-column temporal coherence and/or rhythm-strip
  registration.
- Feasibility gate before building: confirm that the lead-II column-vs-rhythm
  registration (and/or same-column R-wave coherence) actually separates
  `iphone/26` from the clean controls.

**Fallback (plan §9):** if the registration anchor also fails feasibility, the
interpretation-layer errors are not detectable from the current artifacts, which
motivates the data-first path (a larger labelled fidelity set for a learned gate).
