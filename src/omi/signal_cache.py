# One-time preprocessing of the OMI recordings into a memmapped array for training.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.omi.dataset import load_raw_signal
from src.utils.wfdb_helpers import TARGET_LENGTH

LEAD_COUNT = 12
# float16 halves the 4.3 GB the development set would take at float32 and costs
# nothing here: signals are already z-scored, so values sit near [-8, 8].
CACHE_DTYPE = np.float16


def cache_paths(cache_dir: Path, name: str) -> tuple[Path, Path]:
    """Return the (signals, validity) paths for a named cache."""
    return cache_dir / f"{name}_signals.npy", cache_dir / f"{name}_valid.npy"


def build_signal_cache(
    data_dir: Path,
    table: pd.DataFrame,
    cache_dir: Path,
    name: str = "development",
) -> tuple[Path, np.ndarray]:
    """Preprocess every recording once and store it as a memmapped array.

    Re-reading WFDB every epoch dominates training time; the preprocessing is
    deterministic, so it is done once here.

    Args:
        data_dir: Dataset root.
        table: Label table in the order the cache will be indexed by.
        cache_dir: Where to write the cache.
        name: Cache basename.

    Returns:
        (signals_path, valid) — the memmap path and a mask of decodable rows.
    """
    signals_path, valid_path = cache_paths(cache_dir, name)
    if signals_path.exists() and valid_path.exists():
        return signals_path, np.load(valid_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(
        signals_path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=(len(table), LEAD_COUNT, TARGET_LENGTH),
    )
    valid = np.ones(len(table), dtype=bool)
    failures: list[str] = []

    for row_index, record in enumerate(
        tqdm(table.itertuples(index=False), total=len(table), desc="caching signals")
    ):
        try:
            signals[row_index] = load_raw_signal(data_dir, record.ecg_row_record).astype(
                CACHE_DTYPE
            )
        except Exception as error:  # noqa: BLE001 — a bad file must not stop the build
            valid[row_index] = False
            failures.append(f"{record.ecg_row_record}: {error}")

    signals.flush()
    np.save(valid_path, valid)

    if failures:
        print(f"[WARN] {len(failures)} recordings failed to decode:")
        for failure in failures:
            print(f"  {failure}")

    return signals_path, valid


def open_signal_cache(cache_dir: Path, name: str = "development") -> np.ndarray:
    """Open an existing cache read-only, without loading it into RAM."""
    signals_path, _ = cache_paths(cache_dir, name)
    if not signals_path.exists():
        raise FileNotFoundError(f"No signal cache at {signals_path}; build it first")
    return np.load(signals_path, mmap_mode="r")
