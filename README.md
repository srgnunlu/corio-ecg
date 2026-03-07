# Corio ECG

AI-powered paper ECG photograph interpretation pipeline. Converts paper ECG photos into structured diagnostic reports using state-of-the-art foundation models.

## Pipeline

```
Paper ECG Photo (PNG/JPG)
    → ECG-Digitiser (image to 12-lead digital signal)
    → ECGFounder (signal to 150+ diagnoses)
    → LLM Report Engine (diagnoses to structured report)
```

## Key Features

- **Paper ECG input:** Works with photographs or scans of standard 12-lead paper ECGs
- **150+ diagnoses:** Rhythm disorders, conduction abnormalities, ischemic changes, hypertrophy, and more
- **VT/SVT specialization:** Enhanced differentiation using Brugada and Vereckei morphology criteria
- **Structured reports:** Automated generation of clinical ECG reports

## Models Used

| Component | Model | Size | Source |
|-----------|-------|------|--------|
| Digitization | ECG-Digitiser (nnU-Net) | ~475 MB | [GitHub](https://github.com/felixkrones/ECG-Digitiser) |
| Diagnosis | ECGFounder (Net1D CNN) | ~370 MB | [GitHub](https://github.com/PKUDigitalHealth/ECGFounder) |
| Reporting | TBD (open-source LLM preferred) | TBD | TBD |

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
./scripts/download_models.sh

# Download PTB-XL dataset (for evaluation)
./scripts/download_ptbxl.sh
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

# Interpret a single ECG image
python -m src.pipeline.run --image path/to/ecg.png

# Run evaluation on PTB-XL benchmark
python -m src.training.evaluate --dataset ptbxl

# Launch test web UI
python src/web/app.py
```

## Project Structure

```
corio-ecg/
├── src/
│   ├── pipeline/       # Core pipeline (digitize → diagnose → report)
│   ├── training/       # Fine-tuning and evaluation scripts
│   ├── vtsvt/          # VT/SVT specialization (Brugada/Vereckei)
│   ├── utils/          # Helper functions
│   └── web/            # Test web interface
├── configs/            # Training configs and report templates
├── notebooks/          # Jupyter notebooks for experimentation
├── scripts/            # Setup and download scripts
├── data/               # Datasets (gitignored)
├── models/             # Model weights (gitignored)
└── results/            # Metrics, figures, reports
```

## Datasets

| Dataset | Records | Purpose |
|---------|---------|---------|
| [PTB-XL](https://physionet.org/content/ptb-xl/) | 21,799 | Primary benchmark, 71 labels |
| [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/) | 800,000+ | Additional training data |

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
