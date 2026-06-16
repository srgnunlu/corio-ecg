# Session: Segment-Ensemble → Production Diagnosis Path

**Date:** 2026-06-16
**Branch:** `feature/phase3-failure-triage`
**Status:** Done (committed, not pushed)

## TLDR

Segment-ensemble — previously research-only — is now the **default production
diagnosis path** in the web app. Expected gain on PTB-XL round-trip: macro
AUROC ~0.75 (tiled) → ~0.87 (segment-ensemble). Tiled single-pass kept as a
config-selectable fallback. Also fixed a metadata-write bug that silently
failed every `.json` audit sidecar during batch digitization.

## 1. Segment-ensemble production integration (main task)

**Why it works.** Paper ECGs print each lead in a time-shifted column; the
tiled single-pass path feeds all 12 temporally-misaligned (and tiled) leads to
ECGFounder at once, corrupting cross-lead features (axis, bundle blocks). The
segment-ensemble diagnoses each printed paper column independently — its leads
*were* recorded simultaneously — and averages the 150-class probability
vectors. This recovers most of the lost accuracy and was already validated in
`evaluate_roundtrip.py` (macro AUROC ~0.87) but lived only in the eval harness.

**Changes:**
- `src/pipeline/diagnose.py`
  - New `ECGDiagnoser.diagnose_segment_ensemble` / `diagnose_all_segment_ensemble`.
  - Shared module helpers `build_paper_column_signals` + `aggregate_probability_vectors`
    (the canonical column-grouping/aggregation math, reused by the eval).
  - Refactored the single forward pass into `_forward_probabilities` /
    `_collect_results` so tiled and ensemble paths share one code path.
  - Rate-consistency HR adjustment is applied **once** on the aggregated vector
    using the full signal (so the rhythm strip still drives HR heuristics).
- `src/web/app.py`
  - Segment-ensemble is the production default. Env flag
    `CORIO_SEGMENT_ENSEMBLE=0` disables it.
  - `_resolve_segment_layout()` picks the column layout from the user's explicit
    choice, else the digitizer's detected layout; returns `None` (→ tiled
    fallback) for auto-detect / unsupported layouts.
- `src/training/evaluate_roundtrip.py`
  - Now imports the shared helpers from `diagnose.py` (removed the local
    duplicates) so eval and production run **identical** math.
- `src/pipeline/run.py` (clean WFDB CLI) **left unchanged** — its 10 s signals
  are already lead-aligned, so segmenting would only zero out 9 leads per pass
  and hurt accuracy.

**Tests:** `tests/test_diagnose.py` (per-column forward count + averaging, rate
adjustment), `tests/test_web_app.py` (`_resolve_segment_layout`). Full suite:
273 passed, 1 skipped. ruff + mypy clean on changed files.

## 2. Metadata-write bug fix

`scripts/digitize_synthetic_images.py` wrote a per-record `.json` audit sidecar
via `asdict(digitiser.last_info)`. The reprojection-probe fields
`signal_probability` (H×W ink map) and `raw_lines` (full-width pixel traces)
were added to `DigitizeInfo`, but `_json_default` only handled `torch.Tensor`
and `np.generic` — so `json.dumps` raised `TypeError` on the `ndarray` and
every metadata write failed. The `.npy` saved first, so signals looked fine
while the `.json` silently never appeared.

**Fix:** drop the two bulky debug-only arrays before serializing (they have no
place in an audit file) via `_audit_diagnostics()`, and defensively teach
`_json_default` to handle `ndarray`. Regression test added.

## 3. HR-from-rhythm-strip (assessed, NOT changed)

Assessed as **not a safe simple fix** — left for the deferred data-validated
work. `rhythm.py` already prefers Lead II and uses a Pan-Tompkins-style
autocorrelation. The remaining defect is upstream: `_expand_canonical_segments`
(`digitize.py`) crops even the full-width Lead II rhythm strip to a ~2.5 s
column and tiles it, discarding the real 10 s RR sequence. Preserving full-span
rhythm leads there would change core digitization behaviour for *both* HR and
the diagnosis signal and needs real digitized data to validate — too risky to
change blind in this session.

## Commits (branch `feature/phase3-failure-triage`, not pushed)

- `feat: wire segment-ensemble into production diagnosis path`
- `fix: stop bulky DigitizeInfo arrays breaking audit metadata write`
