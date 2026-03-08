# Threshold analysis for round-trip evaluation results
# Shows agreement rates at different probability cutoffs to identify
# how well high-confidence diagnoses are preserved through digitization

import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.pipeline.diagnose import ECGDiagnoser
from src.utils.ecg_labels import NUM_CLASSES
from src.utils.wfdb_helpers import read_ecg_signal
from src.training.evaluate_roundtrip import load_test_fold_ids


def main() -> None:
    model_path = Path("models/ecgfounder/base/12_lead_ECGFounder.pth")
    data_dir = Path("data/raw/ptb-xl")
    sig_dir = Path("data/processed/signals")

    records = load_test_fold_ids(data_dir, max_samples=500)
    diagnoser = ECGDiagnoser(checkpoint_path=model_path)

    baseline_probs: dict[int, np.ndarray] = {}
    roundtrip_probs: dict[str, dict[int, np.ndarray]] = {
        s: {} for s in ["clean", "moderate", "hard"]
    }

    print("Computing baseline probabilities...")
    for ecg_id, path in tqdm(records):
        try:
            sig, _ = read_ecg_signal(path)
            res = diagnoser.diagnose_all(sig)
            p = np.zeros(NUM_CLASSES)
            for r in res:
                p[r.index] = r.probability
            baseline_probs[ecg_id] = p
        except Exception:
            pass

    for scenario in ["clean", "moderate", "hard"]:
        print(f"Computing {scenario} probabilities...")
        for ecg_id, _ in tqdm(records):
            if ecg_id not in baseline_probs:
                continue
            npy = sig_dir / scenario / f"{ecg_id}.npy"
            if not npy.exists():
                continue
            try:
                sig = np.load(npy)
                res = diagnoser.diagnose_all(sig)
                p = np.zeros(NUM_CLASSES)
                for r in res:
                    p[r.index] = r.probability
                roundtrip_probs[scenario][ecg_id] = p
            except Exception:
                pass

    # Threshold analysis table
    print()
    print("=" * 60)
    print("THRESHOLD ANALYSIS — Agreement at different cutoffs")
    print("=" * 60)
    print(f"{'Thresh':>8} {'Clean':>10} {'Moderate':>10} {'Hard':>10}")
    print("-" * 50)

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        row = []
        for scenario in ["clean", "moderate", "hard"]:
            agreements = []
            for ecg_id in roundtrip_probs[scenario]:
                bp = baseline_probs[ecg_id]
                rp = roundtrip_probs[scenario][ecg_id]
                agree = np.mean((bp >= thresh) == (rp >= thresh))
                agreements.append(agree)
            row.append(f"{np.mean(agreements) * 100:.1f}%")
        print(f"{thresh:>8.1f} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # High-confidence analysis
    print()
    print("=" * 60)
    print("HIGH-CONFIDENCE — Baseline prob >= 0.7, flip at 0.5")
    print("=" * 60)
    for scenario in ["clean", "moderate", "hard"]:
        flips = 0
        total_high = 0
        for ecg_id in roundtrip_probs[scenario]:
            bp = baseline_probs[ecg_id]
            rp = roundtrip_probs[scenario][ecg_id]
            high_mask = bp >= 0.7
            total_high += int(np.sum(high_mask))
            flips += int(
                np.sum((bp[high_mask] >= 0.5) != (rp[high_mask] >= 0.5))
            )
        if total_high > 0:
            retention = (1 - flips / total_high) * 100
        else:
            retention = 0.0
        print(
            f"  {scenario:>10}: {total_high} high-conf, "
            f"{flips} flipped -> {retention:.1f}% retained"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
