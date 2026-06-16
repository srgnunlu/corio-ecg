# Corio ECG

Research prototype for AI-assisted paper ECG photograph interpretation.

## Pipeline

```
Paper ECG Photo (PNG/JPG)
    → ECG-Digitiser (image to 12-lead digital signal)
    → ECGFounder (signal to 150+ diagnoses)
    → Planned: structured report engine
```

## Key Features

- **Paper ECG input:** Works with photographs or scans of standard 12-lead paper ECGs
- **150+ diagnoses:** Rhythm disorders, conduction abnormalities, ischemic changes, hypertrophy, and more
- **Clinical measurements:** Heart rate (from the full 10 s rhythm strip), PR / QRS / QT / QTc intervals, and rule-based rhythm classification
- **Structured verdict:** Strict, explainable Normal / Abnormal / Indeterminate headline with the reasons spelled out
- **AI natural-language summary:** Optional Claude (Opus 4.8) prose summary of the findings — Turkish or English, grounded strictly in the structured report (no new diagnoses); needs `ANTHROPIC_API_KEY`
- **Professional web UI:** Modern medical-style Gradio app — drag &amp; drop / camera upload, progress indicator, color-coded result card, interval table, AI-diagnosis confidence bars, and a 12-lead signal plot
- **PDF report:** One-click "Download PDF Report" — verdict, HR/rhythm, intervals, AI diagnoses, original photo, digitized tracing, and disclaimer
- **Quality diagnostics:** Layout, lead activity, Einthoven consistency, and timing checks
- **Research evaluation:** Synthetic round-trip consistency and PTB-XL ground-truth metrics

## Models Used

| Component | Model | Size | Source |
|-----------|-------|------|--------|
| Digitization | ECG-Digitiser (nnU-Net) | ~475 MB | [GitHub](https://github.com/felixkrones/ECG-Digitiser) |
| Diagnosis | ECGFounder (Net1D CNN) | ~370 MB | [GitHub](https://github.com/PKUDigitalHealth/ECGFounder) |
| Measurement | scipy delineation (PR/QRS/QT/QTc) + Pan-Tompkins HR | — | in-repo (`src/measurement/`) |
| Reporting | Deterministic structured report + reportlab PDF + Claude Opus 4.8 narrative | — | in-repo (`src/report/`) |

## Setup

### Prerequisites

- macOS or Linux
- [mise](https://mise.jdx.dev/) (runtime manager)
- ~150 GB free disk space (for datasets)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/corio-ecg.git
cd corio-ecg

# Run setup script (installs Python, creates venv, installs dependencies)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate virtual environment
source .venv/bin/activate

# Download models
python scripts/download_models.py

# Download PTB-XL dataset (for evaluation)
python scripts/download_ptbxl.py

# Download ECGFounder's official PTB-XL target vectors
python scripts/download_ecgfounder_eval_labels.py

# Clone and patch pinned digitizer/image-generation dependencies
python scripts/setup_phase2.py
```

### Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

## Usage

```bash
# Activate environment
source .venv/bin/activate

# Interpret a WFDB signal
python -m src.pipeline.run --signal path/to/record_without_extension

# Run PTB-XL ground-truth evaluation
python -m src.training.evaluate --max-samples 500 --threshold 0.5

# Launch the web UI (http://localhost:7860): upload a photo, get the verdict,
# HR/intervals/rhythm, AI diagnoses, the digitized tracing, and a downloadable PDF.
python -m src.web.app

# Evaluate real phone photos in isolated default and dewarping-retry modes
python scripts/evaluate_real_photos.py --modes default retry --timeout 180

# Use known per-case layouts from data/real-phone/metadata.csv
python scripts/evaluate_real_photos.py --modes default --use-metadata-layouts

# Download the balanced 70-image PMcardio reference subset without fetching the full archive
python scripts/download_pmcardio_reference_subset.py --balanced-count 10

# Compare digitized waveform shape with the matched reference signals
python scripts/evaluate_pmcardio_reference.py --timeout 180

# Compare ECGFounder outputs on matched reference and digitized signals
python scripts/evaluate_pmcardio_diagnosis_drift.py
```

Real-photo inputs belong in `data/real-phone/photos` and remain gitignored.
The evaluator writes anonymous per-photo audit records plus aggregate JSON/CSV
reports to `results/real-phone`. Without a matched PDF, scan, or digital
waveform, these reports measure operational digitization quality only, not
signal fidelity or diagnostic accuracy.

Matched waveform-shape fidelity is measured separately with the GPL-3.0-or-later
[PMcardio ECG Image Database](https://zenodo.org/records/13617673). The current
balanced 70-image subset covers ten matched ECGs across bent, crumpled,
phone-photographed, scanned, and screen-displayed categories. Corio downloads
only the selected ZIP members and writes the benchmark report to
`results/pmcardio-reference`.

### Persistent Gradio Service (macOS)

Use `launchd` if you want the local Gradio UI to survive terminal disconnects.

```bash
# Install the LaunchAgent into ~/Library/LaunchAgents
./scripts/install_gradio_launch_agent.sh

# Start or restart the service
launchctl kickstart -k gui/$(id -u)/com.sergenunlu.corio-ecg.gradio

# Check service status
launchctl print gui/$(id -u)/com.sergenunlu.corio-ecg.gradio

# Stop and unload the service
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.sergenunlu.corio-ecg.gradio.plist
```

Notes:

- The service keeps running even if you close the browser tab.
- If the Python process exits unexpectedly, `launchd` restarts it.
- Logs are written to `/tmp/corio-gradio.log`.
- The service does not start automatically at login; it only runs after manual install/load.
- The LaunchAgent runs as an interactive user service, not a background daemon.
- The local Gradio service disables dewarping retry by default to reduce runaway CPU/memory cases on macOS.

## Project Structure

```
corio-ecg/
├── src/
│   ├── pipeline/       # Core pipeline (digitize → diagnose)
│   ├── measurement/    # HR, PR/QRS/QT/QTc intervals, rhythm analysis
│   ├── report/         # Structured report + PDF generation
│   ├── training/       # Fine-tuning and evaluation scripts
│   ├── vtsvt/          # VT/SVT specialization (Brugada/Vereckei)
│   ├── utils/          # Helper functions
│   └── web/            # Web interface (Gradio)
├── configs/            # Training configs and report templates
├── notebooks/          # Jupyter notebooks for experimentation
├── scripts/            # Setup and download scripts
├── data/               # Datasets (gitignored)
├── models/             # Model weights (gitignored)
└── results/            # Metrics, figures, reports
```

## Current Status

- Working: image digitization, ECGFounder inference, quality diagnostics, synthetic
  round-trip evaluation, and Gradio test UI.
- Verified baseline: local ECGFounder logits match the official implementation;
  full PTB-XL test-fold official macro AUROC is 0.8679.
- Verified round-trip smoke benchmark: 50 records across clean, moderate, and
  hard synthetic photos, with audited pipeline-version metadata.
- Verified balanced matched-reference benchmark: all 70 selected PMcardio images
  digitized; median per-image waveform-shape correlation is 0.823. Calibrated
  output has median RMSE 0.108 mV, median SNR 3.83 dB, and median gain ratio
  0.904.
- Verified matched-reference diagnosis drift: segment-aware mean aggregation
  improves mean ECGFounder cosine consistency from 0.894 to 0.924 and threshold
  agreement from 96.68% to 97.73% versus the tiled representation.
- Verified real-photo operational batch with known paper layouts: all 10 images
  digitized after orientation retry and conservative portrait-screen page crop;
  screen captures still trigger quality warnings.
- In progress: stronger bent/crumpled-paper reconstruction and larger
  class-supported diagnostic evaluation.
- Planned: artifact-aware fine-tuning, VT/SVT specialization, and structured reports.
- This project is for research use only and is not clinically validated.

## Evaluation Notes

Round-trip agreement compares ECGFounder outputs on clean and digitized versions of
the same signal. It measures pipeline consistency, not diagnostic accuracy.

Primary ground-truth evaluation uses ECGFounder's official 150-output PTB-XL target
vectors. A separate semantic-subset metric maps only PTB-XL SCP codes with direct
equivalents in the ECGFounder vocabulary.

Real phone photo collection requirements are documented in
`docs/real-phone-photo-protocol.md`.

Matched-reference methodology and limitations are documented in
`docs/evaluation-methodology.md`.

## Datasets

| Dataset | Records | Purpose |
|---------|---------|---------|
| [PTB-XL](https://physionet.org/content/ptb-xl/) | 21,799 | Primary benchmark, 71 labels |
| [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/) | 800,000+ | Additional training data |
| [PMcardio ECG Image Database](https://zenodo.org/records/13617673) | 6,000+ images | Matched image-to-signal fidelity |

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Inference | Any modern CPU (slow) or 8GB VRAM GPU | Mac Mini M4 / Google Colab T4 |
| Fine-tuning | 16GB VRAM GPU | Vast.ai RTX 4090 / A100 |

## References

- Li et al., "ECGFounder: A Foundation Model for ECG Analysis," NEJM AI, 2024
- Krones et al., ECG-Digitiser, PhysioNet Challenge 2024 Winner
- Wagner et al., "PTB-XL," Scientific Data, 2020
- Brugada et al., "Differential Diagnosis of Wide QRS Complex Tachycardia," Circulation, 1991
- Vereckei et al., "New Algorithm Using Only Lead aVR," Heart Rhythm, 2008

## License

TBD
