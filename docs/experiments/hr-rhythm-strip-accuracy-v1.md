# Experiment: Heart-rate accuracy from the full-duration rhythm strip (v1)

**Date:** 2026-06-16
**Phase:** C.1 (master-roadmap-v1.md) — fix heart rate on paper ECGs.
**Status:** Complete — **success criterion met.** Rhythm-strip HR replaces the
tiled-signal HR in the production diagnosis path.
**Tooling:** `scripts/evaluate_hr_accuracy.py` (re-digitizes each PMcardio image
in an isolated subprocess via `scripts/digitize_one.py`, caches model-input and
rhythm-strip `.npy`, then compares three HR estimates per image).
**Result file:** `results/hr-accuracy/hr_accuracy.json` (n=70 images)

## The problem (recap)

The model is fed a *tiled* diagnosis signal: each lead's printed ~2.5 s segment
repeated to fake 10 s. That tiling destroys the real beat-to-beat (RR) sequence,
so `estimate_heart_rate_bpm` on the tiled signal is unreliable — ~29 % of images
land >20 bpm off the truth (see `diagnosis-tiling-length-mismatch`). The paper
already prints a continuous full-width rhythm lead (usually Lead II) carrying the
genuine 10 s RR sequence, but the digitizer was cropping it during segment
expansion and throwing it away.

## The fix

1. **Preserve the strip.** `digitize.py::_extract_rhythm_strip` keeps full-width
   leads (span ≥ 80 % of the page) uncropped *before* `_expand_canonical_segments`
   crops them to a single column; `_process_rhythm_strip` runs the same
   µV→mV / 500 Hz / highpass / wavelet / z-score chain minus the cropping.
   Surfaced as `DigitizedSignals.rhythm_strip` / `DigitizeInfo.rhythm_strip`.
2. **Estimate HR from RR.** `rhythm.py::estimate_rhythm_hr` runs Pan-Tompkins
   R-peak detection (5–18 Hz bandpass → squared derivative → 120 ms integrator →
   adaptive threshold + refractory gap) over the whole strip and returns
   `60 / median(RR)` — robust to irregular rhythms (AF → ventricular rate),
   brady and tachy alike. Autocorrelation is kept as a noisy-strip fallback.
3. **Wire it through.** `diagnose.py` threads `rhythm_strip` into
   `_apply_rate_consistency_adjustments`; the web app passes
   `debug_info.rhythm_strip`. When no strip exists, it falls back to the old
   tiled estimate — **no regression**, just no improvement on those layouts.

## Headline result (n=70 PMcardio images, ground truth = clean reference strip)

| HR path | median abs err | mean abs err | >20 bpm rate | within 5 bpm | n |
|---|---|---|---|---|---|
| **baseline** — tiled diagnosis signal (production) | 2.0 bpm | 13.8 bpm | **28.6 %** | 61.4 % | 70 |
| **rhythm-strip fix** — RR over full strip | **0.9 bpm** | **3.27 bpm** | **3.0 %** | **83.6 %** | 67 |

**C.1 criterion (median < 5 bpm AND >20 bpm rate < 10 %): PASSED.** The large-error
rate drops from 28.6 % to 3.0 % — a ~10× reduction in the failures that actually
mislead a clinician — and mean error falls from 13.8 to 3.3 bpm (the high baseline
mean is driven by a long tail of octave/tiling blow-ups that the median hides).

## Per-category breakdown (>20 bpm large-error rate)

| category | n | baseline >20 bpm | rhythm-fix >20 bpm | fix n |
|---|---|---|---|---|
| photos_bents | 10 | 40 % | **0 %** | 9 |
| photos_crumbles | 10 | 30 % | **0 %** | 10 |
| photos_doogee | 10 | 40 % | **0 %** | 9 |
| photos_iphone | 10 | 40 % | **0 %** | 9 |
| photos_samsung | 10 | 30 % | **0 %** | 10 |
| photos_scans | 10 | 10 % | 10 % | 10 |
| photos_screens | 10 | 10 % | 10 % | 10 |

The phone-photo categories — exactly where the tiled path failed worst (30–40 %)
— go to **zero** large errors. Clean scans/screens were already decent and stay
roughly flat.

## Residuals

- **Two halving errors (img_43, scans + screens):** reference 99.7 bpm, fix 54.1
  bpm — a classic octave error where the integrator merged adjacent beats or the
  strip's R-peaks were ambiguous and RR doubled. These are the only two >20 bpm
  fix failures; both are on the same source ECG, so it's one hard record, not a
  category-wide weakness.
- **Three images recover no strip (img_10/doogee, img_14/iphone, img_82/bents,
  all 3x4+1R):** the full-width Lead II wasn't digitized cleanly enough to clear
  the 80 % span test, so they fall back to the tiled estimate. No regression, but
  no gain either — a digitizer-fidelity issue, not an HR-method one.

## Actionable conclusions

- **Ship it.** The rhythm-strip HR path is already wired into the production
  diagnosis flow (`web/app.py` → `diagnose_all` / `diagnose_all_segment_ensemble`
  with `rhythm_strip=`). C.1 is done.
- **Next (optional, low priority):** an octave-error guard in `estimate_rhythm_hr`
  (e.g. cross-check median RR against the autocorrelation peak; if they differ by
  ~2×, prefer the denser one) would likely rescue the two img_43 halving cases.
- **Strip recovery (deferred):** the 3 no-strip 3x4+1R images point at digitizer
  fidelity on the rhythm lead, not the HR method — revisit with digitization work.

## Validation

- `tests/test_rhythm.py`: 13 passing, incl. brady/normal/tachy rate recovery,
  irregular-rhythm ventricular rate, Lead-II preference, flat-strip None,
  strip-extraction (full-width preserved / short segments zeroed / no-strip None),
  model-ready processing, and rhythm-strip-overrides-tiled in the diagnosis path.
- Full suite: 285 passed, 1 skipped.
