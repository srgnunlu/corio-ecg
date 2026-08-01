# Extract and cache frozen ECGFounder representations for the OMI dataset.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.omi.dataset import load_median_beat, load_raw_signal
from src.utils.wfdb_helpers import TARGET_LENGTH

# Input variants under comparison. The published baseline used median beats;
# ECGFounder was trained on full 10 s recordings.
INPUT_RAW = "raw"
INPUT_MEDIAN = "median"
INPUT_MODES = (INPUT_RAW, INPUT_MEDIAN)


def _z_score(signal: np.ndarray) -> np.ndarray:
    """Normalise a whole recording, matching the shared WFDB reader."""
    std = signal.std()
    if std < 1e-8:
        return signal - signal.mean()
    return (signal - signal.mean()) / std


def prepare_median_input(median_beat: np.ndarray) -> np.ndarray:
    """Turn a 1 s median beat into a 10 s input the backbone accepts.

    The beat is repeated to fill the expected length. Rhythm information in the
    result is synthetic, which is acceptable here because OMI is a morphological
    finding (ST/T shape), not a rhythm one — but it does mean rhythm-dependent
    features from this variant are meaningless.

    Args:
        median_beat: Array of shape (12, 500) in mV.

    Returns:
        Z-scored array of shape (12, 5000).
    """
    repeats = int(np.ceil(TARGET_LENGTH / median_beat.shape[1]))
    tiled = np.tile(median_beat, (1, repeats))[:, :TARGET_LENGTH]
    return _z_score(tiled)


def extract_features(
    model: torch.nn.Module,
    device: torch.device,
    data_dir: Path,
    table: pd.DataFrame,
    input_mode: str = INPUT_RAW,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen backbone over a label table and return pooled features.

    A handful of recordings in the dataset fail to decode — the paper itself
    reports 99.99% signal-duration compliance. Those get a zero feature row and
    a False validity flag rather than aborting the run; callers drop them.

    Args:
        model: Loaded Net1D, already in eval mode.
        device: Device to run on.
        data_dir: Dataset root.
        table: Label table with `ecg_row_record` / `ecg_med_record` columns.
        input_mode: "raw" for 10 s recordings, "median" for tiled median beats.
        batch_size: Records per forward pass.

    Returns:
        (features, valid) — features has shape (len(table), feature_dim) and
        valid is a boolean mask of the rows that decoded successfully.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")

    records = list(table.itertuples(index=False))
    features: list[np.ndarray] = []
    valid = np.ones(len(records), dtype=bool)
    failures: list[str] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc=f"features:{input_mode}"):
            chunk = records[start : start + batch_size]
            signals = []
            for offset, record in enumerate(chunk):
                try:
                    if input_mode == INPUT_RAW:
                        signals.append(load_raw_signal(data_dir, record.ecg_row_record))
                    else:
                        signals.append(
                            prepare_median_input(
                                load_median_beat(data_dir, record.ecg_med_record)
                            )
                        )
                except Exception as error:  # noqa: BLE001 — one bad file must not stop the run
                    valid[start + offset] = False
                    failures.append(f"{record.ecg_row_record}: {error}")
                    signals.append(np.zeros((12, TARGET_LENGTH), dtype=np.float32))
            batch = torch.tensor(np.stack(signals), dtype=torch.float32, device=device)
            pooled = model.forward_features(batch)
            features.append(pooled.detach().cpu().numpy())

    if failures:
        print(f"\n[WARN] {len(failures)} recordings failed to decode and were excluded:")
        for failure in failures[:10]:
            print(f"  {failure}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    return np.concatenate(features, axis=0), valid


def cached_features(
    cache_path: Path,
    build: callable,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached (features, valid), computing and storing them on first use.

    Feature extraction over the full dataset costs minutes of GPU time and the
    backbone is frozen, so the result is stable and worth keeping on disk.
    """
    valid_path = cache_path.with_name(f"{cache_path.stem}_valid.npy")
    if cache_path.exists() and valid_path.exists():
        return np.load(cache_path), np.load(valid_path)

    features, valid = build()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    np.save(valid_path, valid)
    return features, valid
