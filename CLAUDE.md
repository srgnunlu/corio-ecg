# Corio ECG - Project Instructions

## Project Overview
Paper ECG photograph interpretation AI pipeline. Takes a photo of a paper ECG, digitizes it to signal, runs 150+ diagnosis classification, and generates a structured report via LLM.

**Pipeline:** Photo → ECG-Digitiser (image→signal) → ECGFounder (signal→diagnosis) → LLM (diagnosis→report)

## Tech Stack
- **Language:** Python 3.12
- **Environment:** mise (Python) + venv
- **ML Framework:** PyTorch >= 2.4.0
- **Core Models:**
  - ECG-Digitiser: nnU-Net 2D segmentation (~475 MB) - BSD-2 license
  - ECGFounder: Net1D CNN, 76.3M params (~370 MB) - NEJM AI 2024
- **Data Format:** WFDB (.dat + .hea), 12-lead, 500 Hz
- **Datasets:** PTB-XL (21K records), MIMIC-IV-ECG (800K+ records)
- **Test UI:** Gradio or Streamlit (test phase only)

## Key Conventions

### File Organization
- Max 250 lines per file (hard limit 400)
- One component/module per file
- Comments in English, explain WHY not WHAT
- Every file starts with a brief purpose comment

### Naming
- Files: `snake_case.py`
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Descriptive names: `patient_diagnosis` not `pd`, `ecg_signal` not `sig`

### Code Style
- Type hints on all function signatures
- Docstrings on public functions (Google style)
- No hardcoded paths — use configs/constants
- All paths relative to project root or configured via environment variables

## Data & Model Files
- **NEVER commit** model weights, datasets, or large binary files to git
- Models go in `models/` (gitignored)
- Data goes in `data/` (gitignored)
- Use scripts in `scripts/` to download models and data

## Project Phases (from SPEC.md)
1. **Phase 1:** Pipeline setup (Digitiser + ECGFounder working)
2. **Phase 2:** Digitization robustness (fine-tune for noisy signals)
3. **Phase 3:** General improvement (add MIMIC-IV-ECG data)
4. **Phase 4:** VT/SVT specialization (Brugada/Vereckei criteria)
5. **Phase 5:** LLM report generation
6. **Phase 6:** Test web UI
7. **Phase 7:** Academic publications

## Important Paths
- `SPEC.md` — Full project specification
- `src/pipeline/` — Core pipeline code (digitize, diagnose, report)
- `src/training/` — Fine-tuning scripts
- `src/vtsvt/` — VT/SVT specialization module
- `configs/` — Training hyperparameters and report templates
- `notebooks/` — Jupyter notebooks for experimentation
- `results/` — Metrics, figures, generated reports

## Security
- No patient data (PHI/PII) in this repo — only open-source datasets
- API keys in `.env` file (gitignored)
- Never log or print sensitive information

## Running the Pipeline
```bash
# Activate environment
source .venv/bin/activate

# Run diagnosis on a WFDB signal file
python -m src.pipeline.run --signal data/raw/ptb-xl/records500/00000/00001_hr --threshold 0.5

# Run baseline evaluation on PTB-XL (quick test with 20 samples)
python -m src.training.evaluate --max-samples 20

# Download model and data
python scripts/download_models.py
python scripts/download_ptbxl.py
```

## Session History
- Session summaries are saved to `docs/sessions/` after each work session
- Use `/session-summary` or say "oturumu özetle" to create one
- Read previous session summaries to understand project history and pick up where left off

## Development Notes
- Mac Mini M4 (24GB RAM) is sufficient for inference and light fine-tuning
- Heavy fine-tuning: use Vast.ai RTX 4090 ($0.28/hr) or Google Colab
- ECGFounder needs ~1 GB VRAM, ECG-Digitiser needs ~4 GB VRAM
- Both models fit on a single GPU with room to spare
