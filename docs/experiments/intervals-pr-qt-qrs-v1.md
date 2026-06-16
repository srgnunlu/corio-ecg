# Experiment: PR / QRS / QT / QTc interval measurement (v1)

**Date:** 2026-06-17
**Phase:** C.2 (master-roadmap-v1.md) — interval measurements (greenfield).
**Status:** Complete — module implemented, unit-validated on synthetic
known-interval ECGs, wired into the production diagnosis path and web UI.
**Modules:** `src/measurement/delineation.py` (per-beat fiducial detection),
`src/measurement/intervals.py` (orchestration, aggregation, QTc, interpretation).
**Tests:** `tests/test_intervals.py` (10 tests).

## Goal

Report the intervals a clinician expects beside the diagnosis probabilities:

- **PR** — P onset → QRS onset. Normal 120–200 ms; >200 ms ⇒ first-degree AV block.
- **QRS** — QRS onset → offset. Normal <120 ms; ≥120 ms ⇒ bundle branch block.
- **QT / QTc** — QRS onset → T end. QTc = QT corrected for rate. Normal QTc <450 ms
  (male) / <460 ms (female); prolongation is clinically critical (drug effect,
  sudden-death risk).

## Design decisions

### No new dependency (scipy-only delineation)

The roadmap floated **NeuroKit2** but flagged it as a dependency needing approval,
and the global rule is "new dependency → tell me first." NeuroKit2 is MIT-licensed
and maintained, so it remains a viable future upgrade — but for v1 the delineation
is built on `scipy.signal` alone, matching the existing `rhythm.py` (Pan-Tompkins,
pure scipy). Zero new dependencies, fully testable, and consistent with the
codebase. If clinical validation later shows the simple delineator is the
accuracy bottleneck, swapping in NeuroKit2's wavelet delineator is the obvious
next step.

### Measurement source: rhythm strip first

Measurements run on **Lead II of the full-duration rhythm strip** when available
(the genuine 10 s strip recovered in C.1), because QTc needs the *real* RR
sequence — the tiled diagnosis signal repeats a ~2.5 s segment and carries no true
RR. When no strip exists the code falls back to the tiled signal: intra-beat
morphology (PR/QRS/QT) is still valid there, only the RR (and thus QTc) is less
reliable.

### Algorithm (per beat, median across beats)

1. **R peaks** — Pan-Tompkins envelope (5–18 Hz bandpass → squared derivative →
   120 ms integrator → adaptive threshold + refractory), refined to the largest
   absolute deflection within ±50 ms.
2. **QRS onset/offset** — outermost crossings of 15 % of local QRS energy
   (|smoothed derivative|). The energy is bi-lobed with a dip *at* the R apex, so
   boundaries are taken as the span of the threshold-exceeding region, not by
   walking outward from R (which stalls in the apex dip).
3. **P onset** — the deflection *closest* to QRS onset (not the largest) above
   10 % of the QRS-region swing, within a window capped by 0.45·RR — this avoids
   locking onto the tail of the preceding T wave at fast rates (P-on-T). Returns
   none when no organized P exists (e.g. atrial fibrillation).
4. **T end** — classic tangent (Lepeschkin) method: find the steepest descending
   slope after the T peak, intersect that tangent with the baseline.
5. **Aggregation** — median of per-beat measurements (robust to an occasional
   missed P or T); each measurement is dropped if outside physiological bounds.
6. **QTc** — Bazett (QT/√RR) and Fridericia (QT/∛RR). Fridericia is preferred at
   rate extremes (HR <60 or >100) or irregular rhythm (RR CV >0.15, e.g. AF),
   where Bazett over/under-corrects; otherwise Bazett.

## Validation (synthetic known-interval ECGs)

Triangular-wave ECGs with explicit fiducials give exact ground truth. Recovered
intervals across rates:

| Truth (bpm / PR / QRS / QT) | Measured PR | Measured QRS | Measured QT | QTc formula |
|---|---|---|---|---|
| 60 / 160 / 90 / 380 | 148 | 98 | 374 | Bazett |
| 75 / 200 / 120 / 400 | 188 | 128 | 396 | Bazett |
| 120 / 140 / 90 / 320 | 128 | 98 | 320 | Fridericia |
| 50 / 180 / 100 / 420 | 168 | 108 | 414 | Fridericia |

All within ~14 ms of truth (test tolerances: PR/QRS ±30 ms, QT ±40 ms). Bazett
QTc equals QT at exactly 60 bpm (RR = 1 s) as a sanity check. Note the small
systematic offsets (PR ≈ −12 ms, QRS ≈ +8 ms) are threshold-definition artifacts,
well inside clinical tolerance.

## Integration

- `ECGDiagnoser` computes intervals as a **best-effort** side-effect inside
  `_apply_rate_consistency_adjustments` (the single point where both the signal
  and rhythm strip are in hand) and stores them on `last_interval_measurements`.
  Any delineation error is caught and logged — diagnosis never breaks.
- The web app renders an **Interval Measurements** card above the critical
  findings: PR / QRS / QT / QTc with abnormality flags, measured lead, beat count,
  confidence, and a "research use — verify against the tracing" disclaimer.

## Limitations / next steps

- **Validated on synthetic signals only.** Real-photo concordance against PTB-XL
  machine measurements or expert reads is **not yet measured** (roadmap C.2
  success criterion: QTc ±20 ms concordance). That needs labelled interval data —
  same lead-time bottleneck as the rest of Phase B/C.
- Triangular test waves are an idealization; real QRS/T morphology will stress the
  tangent method and energy thresholds more.
- No per-lead fusion yet (measures a single best lead). Multi-lead "global"
  delineation (earliest onset / latest offset across leads) is the standard
  clinical convention and a natural v2.
- AF QTc uses Fridericia automatically via the RR-irregularity branch, but PR is
  correctly suppressed (no organized P) rather than reported.
