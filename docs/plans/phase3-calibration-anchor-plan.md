# Phase 3 Implementation Plan: Calibration / Assignment Physical Anchor

**Date:** 2026-06-16
**Status:** Plan — implementation to continue in a later session
**Author handoff:** Sergen + Claude
**Predecessors (all rejected):**
`phase3-stability-probes-rejected`, `phase3-goldberger-consistency-v1`,
`phase3-reprojection-fidelity-v1`

## 1. Why this target (the whole point)

Four reference-free probes have failed (shadow stability, geometric perturbation,
Goldberger redundancy, image-space re-projection). They failed for **one shared
root cause**: the surviving false accepts (`doogee/74`, `iphone/26`) are
self-consistent *and* faithful to the printed ink, but wrong in the
**interpretation layer** — absolute grid-scale calibration (mm/mV, mm/s) and/or
lead assignment. These errors are self-consistent by construction, so no internal
or self-overlay check can see them. See `docs/experiments/phase3-reprojection-fidelity-v1.md`.

The only thing that can catch them is an **absolute physical anchor**:
- **Calibration:** the printed 1 mV / 10 mm reference pulse, and the grid spacing,
  are external ground truth for the amplitude/time scale.
- **Assignment:** the printed lead-name labels are external ground truth for which
  trace is which lead.

This plan builds and pilots both, tune-only, with the locked 60-group test split
staying closed.

## 2. Key facts established this session (do not re-derive)

- **Grid-based scale** `avg_pixel_per_mm` comes from
  `PixelSizeFinder` (external `src/model/pixel_size_finder.py`): it autocorrelates
  the **grid probability map** and finds grid-line spacing assuming **5 mm between
  grid lines**. This is the *only* scale estimate today and is already a gate
  feature.
- The **calibration pulse is currently ignored** everywhere (no handling in the
  external repo or our code).
- `canonical_lines` / `raw_lines` x-axis = aligned-image **pixel columns**;
  `canonical_lines` values are **mV** (already calibrated with the grid scale).
- We already expose (vendored patch) `signal_probability` (H×W ink map) and
  `extraction_crop_x0`. `raw_lines` (pixel-Y traces) and `detected_lead_names`
  are exposed; **lead-label pixel positions are NOT** yet exposed.
- Lead order: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1..V6=6..11.
- Tune false accepts: `doogee/74` (true corr 0.894), `iphone/26` (true corr
  0.515). Controls = highest-fidelity `3x4+1R`/`3x4+3R`.
- Digitizing is expensive (~6 s, multi-GB); always isolate per image in a
  subprocess (`scripts/*_one.py` pattern, `--no-retries`).

## 3. Phase 0 — Failure triage (do this FIRST, ~1h, no new code paths)

We do not yet know whether the false accepts fail by **calibration scale** or by
**lead assignment**. Decide before building, using data already on disk.

- **Task 0.1:** For `iphone/26` and `doogee/74`, load the tune reference signal
  and the digitized calibrated signal (re-digitize via `reproject_one`-style
  worker, or reuse `pmcardio_reference_fidelity.json` `per_lead`).
- **Task 0.2:** Classify the error:
  - **Global amplitude scale error** → digitized ≈ `k ·` reference for a single
    constant `k ≠ 1` across leads (calibration-scale). Check by regressing
    digitized vs reference per lead; a consistent slope ≠ 1 with high R² = scale
    error.
  - **Lead permutation** → lead `i` of the digitized signal correlates best with
    a *different* reference lead `j ≠ i` (assignment error). Check the 12×12
    cross-correlation matrix; off-diagonal argmax = permutation.
  - **Morphology/extraction error** → neither; low correlation everywhere even
    after best scale/permutation (image-overlay would have caught this — it
    didn't, so unlikely).
- **Deliverable:** `docs/experiments/phase3-failure-triage-v1.md` stating, per
  false accept, which track (calibration vs assignment) is responsible. This
  decides the priority order of Tracks A and B below.

> If `iphone/26` is a **scale** error, Track A is the primary win. If it is a
> **permutation** error, Track B is. Build the primary track first, then the
> other.

## 4. Track A — Calibration-pulse / grid-scale anchor

### A.1 Hypothesis
A faithful digitization reproduces the printed 1 mV reference pulse at ≈ 1.0 mV;
a calibration-scale error reproduces it at ≠ 1.0 mV. Equivalently, an independent
pulse-derived px/mm disagrees with the grid-derived `avg_pixel_per_mm`.

### A.2 Detection approach (recommended: signal-space, with image-space fallback)

**Primary — signal-space pulse amplitude (Option A):**
- The calibration pulse is a rectangular, flat-topped step (≈ 1 mV high, ≈ 200 ms
  wide) usually at the **start of each lead row**.
- Search the first ~0.4 s of each canonical lead (and/or the rhythm strip) for a
  rectangular plateau: a fast rising edge, a stable flat top of width within
  [~120, ~300] ms, a fast falling edge, sitting on the isoelectric baseline.
- Measure the plateau height in **mV** from `canonical_lines` (already
  calibrated). Aggregate across leads/rows that contain a detectable pulse:
  `calibration_pulse_amplitude_mv` (median), `calibration_pulse_deviation =
  |median_amplitude - 1.0|`, `calibration_pulse_confidence` (how many leads had a
  clean pulse / fit quality).
- **Feature meaning:** large deviation from 1.0 mV ⇒ untrustworthy amplitude
  calibration ⇒ flag/reject.

**Fallback — image-space pulse height (Option B), if Phase 0/feasibility shows the
pulse is cropped out of the extracted signal:**
- Expose the **aligned image** (or aligned grid/signal prob) from the wrapper
  (one more vendored-patch line) and detect the rectangular pulse in pixel space
  at the row left margins; height_px / 10 mm = pulse px/mm_y.
- Compare to grid `avg_pixel_per_mm`: `scale_disagreement_ratio =
  pulse_px_per_mm / grid_px_per_mm`. Far from 1.0 ⇒ calibration suspect.

### A.3 Feasibility gate (mandatory before full build)
Digitize 3–4 tune records and **inspect the first 0.4 s of each lead** (plot
saved to `results/.../calibration/feasibility/`). Confirm a rectangular pulse is
actually present and survives extraction. If absent in most records (smartphone
crops, no printed pulse), **switch to Option B** or **descope Track A** and lead
with Track B. Record the finding before writing the detector.

### A.4 Module / files
- `src/quality/calibration_pulse.py` — pure, testable:
  - `detect_calibration_pulse(lead_signal, sample_rate) -> PulseFit | None`
    (edge/plateau detection; returns amplitude_mv, width_ms, fit confidence).
  - `calibration_anchor(canonical_mv, sample_rate) -> CalibrationAnchor`
    (aggregate amplitude, deviation, confidence, n_pulses_found).
- `scripts/calibration_one.py` — isolated worker: digitize one image, run the
  anchor on `canonical_millivolts` (expose it via `digitize_with_calibrated`
  outputs / `last_info`), print JSON.
- `scripts/calibration_pilot.py` — orchestrate over 2 false accepts + 10 controls,
  report separation margin (residual = deviation; false accepts should exceed
  controls ⇒ positive margin).
- `tests/test_calibration_pulse.py` — synthetic leads with a known 1.0 mV pulse
  (high confidence, low deviation), a 1.3 mV pulse (deviation 0.3), a no-pulse
  flat/echo lead (returns None / low confidence), width-out-of-range rejection.

### A.5 Acceptance
- Feasibility passes (pulse detectable in ≥ ~70% of in-scope tune leads).
- Tune separation margin **positive** (false accept deviations above all control
  deviations), and specifically `iphone/26` is flagged.

## 5. Track B — Lead-assignment verification

### B.1 Hypothesis
A wrong lead assignment places a trace under the wrong canonical lead. The printed
lead-name labels (Lead Name U-Net) are independent ground truth for which lead
sits in which grid cell.

### B.2 Instrumentation
- Expose **lead-label pixel positions** from the identifier (`_extract_lead_points`
  already computes `(name, x, y)` for detected labels; today only the names leave
  via `detected_lead_names`). Add the positions to the wrapper return (vendored
  patch) — small, in the same spirit as the re-projection instrumentation.

### B.3 Feature
- Map each detected label position to its layout grid cell (the same
  `column_edges` / row logic used in `_canonicalize_lines`), and check whether the
  label name matches the canonical lead the layout assigned to that cell.
- `lead_assignment_match_fraction` = matched cells / detected labels.
  `lead_assignment_conflicts` = count of mismatches. Low match ⇒ assignment
  untrustworthy.

### B.4 Module / files
- `src/quality/lead_assignment_check.py` (pure, testable):
  `verify_lead_assignment(detected_labels_with_pos, layout_name, image_width) ->
  AssignmentCheck`.
- Reuse the existing `scripts/*_one.py` + `*_pilot.py` pattern; add tests with
  synthetic label sets (all-correct → 1.0; one swap → conflict).

### B.5 Acceptance
- Tune separation margin positive on `lead_assignment_match_fraction`
  (false accepts lower than controls), catching whichever false accept Phase 0
  attributed to assignment.

## 6. Integration plan (only if a track succeeds on tune)

1. Add the winning feature(s) to `configs/quality_feature_contract_v1.yaml`
   (move `calibration_pulse_confidence` out of `planned_unavailable_features`;
   add e.g. `calibration_pulse_deviation_mv`, `lead_assignment_match_fraction`
   with value_type/min/max/unit/missing_policy/description).
2. Wire population in `src/pipeline/digitize.py::_populate_diagnostics`
   (compute from `canonical_millivolts` / exposed label positions; store on
   `DigitizeInfo`).
3. Add thresholds to `configs/quality_gate_v1.yaml` and a reason code in
   `src/quality/models.py` (`UNTRUSTWORTHY_CALIBRATION`,
   `LEAD_ASSIGNMENT_CONFLICT`); branch in `src/quality/gate.py`.
4. Re-freeze the gate (`scripts/evaluate_quality_gate.py` on tune), confirm no
   regression in false-reject rate, THEN — and only once — run the locked
   60-group test split.

## 7. Risks & open questions

- **Pulse presence (highest risk).** Smartphone photos of clinical ECGs may crop
  the pulse, or the printout may lack one. Phase A.3 feasibility gates this.
- **Pulse location varies** (per-row left margin vs once per page; staircase vs
  single step). Detector must be tolerant and report confidence, not assume.
- **Grid scale may itself be the wrong reference.** If `avg_pixel_per_mm` is
  wrong because the grid autocorrelation locked onto the wrong harmonic, the pulse
  is the *more* trustworthy anchor — prefer the absolute 1.0 mV check over the
  ratio.
- **Underpowered tune set.** Two false accepts cannot establish a rate; a positive
  tune margin is necessary, not sufficient. Keep claims internal; the locked test
  is one-shot.
- **External patch drift.** All wrapper changes go through
  `vendor/patches/open-ecg-digitizer.patch` (regenerate via `git -C
  external/open-ecg-digitizer diff | sed 's/^ $//' > <patch>`; verify with
  `git apply --reverse --check`).

## 8. Ordered task list (pick-up checklist)

- [ ] **0.** Phase 0 failure triage (`phase3-failure-triage-v1.md`); decide A vs B priority.
- [ ] **A.3** Calibration-pulse feasibility inspection on 3–4 tune records.
- [ ] **A.4** `src/quality/calibration_pulse.py` + tests.
- [ ] **A** `scripts/calibration_one.py` + `scripts/calibration_pilot.py`; expose
      `canonical_millivolts` to the worker.
- [ ] **A.5** Run pilot; write `docs/experiments/phase3-calibration-anchor-v1.md`.
- [ ] **B.2** Expose lead-label positions (vendored patch + regenerate).
- [ ] **B.4** `src/quality/lead_assignment_check.py` + tests + worker + pilot.
- [ ] **B.5** Run pilot; write `docs/experiments/phase3-lead-assignment-v1.md`.
- [ ] **6.** If a track succeeds on tune: contract + gate integration, re-freeze.
- [ ] Commit per step (`feat:`), push only when asked. Keep the 60-group test LOCKED.

## 9. Definition of done (this initiative)

Either: a feature with a **positive tune separation margin** that flags the
relevant false accepts, integrated into a re-frozen gate and validated once on the
locked test — or a documented rejection establishing that the interpretation-layer
errors are not detectable from the available artifacts, motivating the data-first
path (larger labelled fidelity set, per PMcardio/PhysioNet).
