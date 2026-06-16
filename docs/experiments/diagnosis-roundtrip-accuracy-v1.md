# Experiment: End-to-end diagnosis accuracy on paper ECGs (v1)

**Date:** 2026-06-16
**Status:** Complete — the definitive image→digitize→diagnose accuracy eval.
**Tooling:** `scripts/generate_synthetic_images.py` (render PTB-XL → paper image,
"moderate" difficulty, n=300) → `scripts/digitize_synthetic_images.py` → existing
`src/training/evaluate_roundtrip.py` (scored vs PTB-XL ground-truth labels).
**Result file:** `results/metrics/roundtrip_comparison.json`

## Why synthetic images

Measuring diagnosis *accuracy* needs an image **and** a ground-truth diagnosis for
the same record. PMcardio has real photos but no diagnosis labels; PTB-XL has
labels but no images. So we render PTB-XL's real signals to paper-style images,
digitize, diagnose, and compare to PTB-XL's labels. Caveat: synthetic images are
cleaner than real phone photos, so these numbers are an **optimistic upper bound**.

## Headline result (semantic SCP subset, n=300, 21 classes)

| feed path | macro AUROC | micro F1 |
|---|---|---|
| **baseline** — clean 10 s signal (model ceiling) | 0.905 | 0.666 |
| **current production** — tile each lead, feed whole | **0.754** | **0.371** |
| **segment-ensemble** — diagnose each column separately, aggregate | **0.871** | 0.522 |

Per-class F1 (the real story):

| class | baseline | current (tiled) | segment-ensemble |
|---|---|---|---|
| SINUS RHYTHM | 0.88 | 0.87 | 0.88 |
| SINUS TACHYCARDIA | 0.84 | **0.09** | **0.78** |
| ATRIAL FIBRILLATION | 0.74 | **0.11** | **0.65** |
| NORMAL ECG | 0.76 | **0.00** | 0.31 |
| PREMATURE VENTRICULAR COMPLEXES | 0.93 | 0.20 | 0.45 |

## Interpretation — the diagnosis collapse is mostly an inference-strategy bug

1. **The production path destroys diagnosis.** The app (`web/app.py:128` →
   `diagnose_all` → `_align_leads_for_model`) feeds the model one signal where
   each lead's ~2.5 s printed segment is tiled to fake 10 s and all 12 leads are
   crammed together. Result: NORMAL ECG → **0.00**, SINUS TACHYCARDIA → **0.09**,
   AFib → **0.11**. This matches what users see: even simple calls fail.

2. **The model and the digitized morphology are mostly fine.** Feeding the *same*
   digitized signals via **segment-ensemble** — each printed column as its own
   sparse 12-lead input, leads in their true positions, no tiling, outputs
   aggregated — recovers macro AUROC from 0.754 to **0.871** (ceiling 0.905) and
   rescues SINUS TACHYCARDIA (0.09→0.78) and AFib (0.11→0.65).

3. **Attribution of the ~0.15 AUROC loss:** ~0.12 is **inference strategy**
   (tiling + whole-signal feed) — fixable in software — and only ~0.03 is genuine
   digitization fidelity. The earlier tiling-only isolation
   (`diagnosis-tiling-impact-v1.md`) already showed the tiling feed is the culprit;
   this confirms it end-to-end against real labels.

4. **Still weak even with segment-ensemble:** NORMAL ECG (0.31) and PVC (0.45).
   NORMAL needs all 12 leads judged together (hard when split by column); PVC and
   other ectopy need the full 10 s rhythm strip. These are the residual targets.

## Actionable conclusions

- **Highest-value fix (software, validated):** switch the production diagnosis
  path from `_align_leads_for_model` (tile+whole) to **segment-ensemble**
  per-column inference, which already exists in `evaluate_roundtrip.py` but is not
  wired into `diagnose.py` / the web app. Expected: macro AUROC 0.75 → ~0.87.
- **HR fix (separate, queued):** estimate HR from the genuine 10 s rhythm strip,
  not the tiled leads (see `diagnosis-tiling-length-mismatch`).
- **Residual (harder):** NORMAL-ECG confirmation and ectopy (PVC/PAC) need a
  whole-12-lead or rhythm-strip-aware path; revisit after the segment-ensemble
  switch.

## Known issue surfaced during this run

`scripts/digitize_synthetic_images.py` reported "Failed: 250" because its audit
**metadata** write (`_json_default`) cannot serialise an ndarray field in the
diagnostics dict. The **signals themselves digitized correctly** (all 300 valid
(12,5000), z-scored) — only the `.json` sidecar failed, and the eval uses only the
`.npy`. The serializer should handle `np.ndarray`, not just `np.generic`/Tensor.
