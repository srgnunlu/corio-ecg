# Phase 2 Design: Digitization Robustness

**Date:** 2026-03-08
**Phase:** 2 — Digitization Robustness
**Goal:** Measure and mitigate diagnostic accuracy loss when ECG signals pass through the paper-to-digital round-trip (signal → synthetic image → digitized signal → diagnosis).

**Potential Paper:** "Impact of ECG Digitization Artifacts on AI Diagnostic Accuracy: Evaluation and Mitigation Using Foundation Models"

---

## 1. Overview

ECGFounder was trained on clean digital signals. In production, signals come from paper ECG photographs, introducing digitization artifacts (noise, distortion, resolution loss). Phase 2 quantifies this performance gap and prepares for fine-tuning to close it.

### Round-Trip Pipeline

```
PTB-XL signals (clean ground truth)
    |
    +---> ECGFounder --> Diagnosis (baseline, from Phase 1)
    |
    +---> ECG-Image-Kit --> Synthetic paper ECG image
                |
                +---> ECG-Digitiser --> Digitized signal
                            |
                            +---> ECGFounder --> Diagnosis (digitized)
                                        |
                                        +---> Compare with baseline
```

---

## 2. Components

### 2.1 ECG-Digitiser Integration

- **Source:** https://github.com/felixkrones/ECG-Digitiser
- **Location:** `external/ecg-digitiser/` (git clone, gitignored)
- **Model:** nnU-Net 2D segmentation, `models/digitiser/M3/` (~475 MB via Git LFS)
- **Key detail:** Uses a custom nnU-Net fork (fixes RGB PNG bug in official nnU-Net)
- **Integration:** Python API import (not subprocess CLI calls)
- **Update:** `src/pipeline/digitize.py` — replace placeholder with real wrapper

### 2.2 Synthetic Image Generation (ECG-Image-Kit)

- **Source:** https://github.com/alphanumericslab/ecg-image-kit
- **Location:** `external/ecg-image-kit/` (git clone, gitignored)
- **Module:** `src/utils/ecg_render.py` — wrapper around ECG-Image-Kit
- **Three difficulty levels:**
  - **Clean:** Grid + signal, no distortions
  - **Moderate:** Light noise, small rotation, paper texture
  - **Hard:** Wrinkles, stains, perspective distortion, handwritten notes
- **Output:** `data/processed/images/{clean,moderate,hard}/`

### 2.3 Round-Trip Evaluation

- **Script:** `src/training/evaluate_roundtrip.py`
- **Dataset:** PTB-XL test fold (strat_fold == 10, ~2000 records)
- **Comparison scenarios:**

| Scenario | Signal Source | Expected Performance |
|----------|-------------|---------------------|
| Baseline | PTB-XL clean signal | Best (from Phase 1) |
| Clean roundtrip | Clean image → digitized | Slight drop |
| Moderate roundtrip | Moderate distortion → digitized | Noticeable drop |
| Hard roundtrip | Heavy distortion → digitized | Significant drop |

- **Signal quality metrics:** SNR (dB), Pearson correlation (per-lead)
- **Diagnosis metrics:** AUROC (macro + per-class), sensitivity, specificity
- **Output:** `results/metrics/roundtrip_comparison.json` + comparison figures

---

## 3. File Structure

```
external/
└── ecg-digitiser/                  <- git clone (gitignored)
└── ecg-image-kit/                  <- git clone (gitignored)

src/
├── pipeline/
│   └── digitize.py                 <- placeholder -> real ECG-Digitiser wrapper
├── utils/
│   └── ecg_render.py               <- NEW: signal -> synthetic paper ECG image
└── training/
    ├── evaluate.py                  <- existing baseline evaluator
    └── evaluate_roundtrip.py        <- NEW: round-trip comparison

scripts/
└── generate_synthetic_images.py     <- NEW: batch image generation script

data/processed/
├── images/{clean,moderate,hard}/    <- synthetic images
└── signals/{clean,moderate,hard}/   <- digitized signals

results/metrics/
└── roundtrip_comparison.json        <- comparison results
```

---

## 4. Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| ECG-Digitiser location | `external/` git clone | Easy updates, isolated from our code |
| ECG-Image-Kit location | `external/` git clone | Same rationale |
| venv strategy | Single venv (TF + PyTorch) | Simpler pipeline, fallback to separate if conflicts |
| nnU-Net version | ECG-Digitiser's custom fork | Official nnU-Net has RGB PNG bug |
| Image generation tool | ECG-Image-Kit (not PTB-XL-Image-17K) | Need control over augmentation parameters for paper |
| Difficulty levels | 3 (clean/moderate/hard) | Systematic analysis of degradation curve |

---

## 5. Dependencies to Add

```
# Already in pyproject.toml [digitiser] optional group:
tensorflow>=2.14.0
scikit-image>=0.23.2
imgaug>=0.4.0
imutils>=0.5.4

# May need to add:
imageio>=2.34.1
```

---

## 6. Success Criteria

- ECG-Digitiser produces valid 12-lead WFDB output from synthetic images
- Round-trip SNR > 15 dB on clean images
- Performance gap between baseline and round-trip is quantified with confidence intervals
- All metrics saved in reproducible format for the paper

---

## 7. Out of Scope (Phase 2)

- Fine-tuning ECGFounder for digitization noise (may start if time permits, but main fine-tuning is later)
- Real clinical paper ECG testing (no real photos yet)
- ECG-Digitiser fine-tuning (Phase 2 uses pre-trained weights only)
