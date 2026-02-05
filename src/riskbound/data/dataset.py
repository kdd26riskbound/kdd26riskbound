from __future__ import annotations

from riskbound.settings import DATA_DIR
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import json

def load_data(market: str, base_dir: Optional[str | Path] = None) -> Tuple[np.ndarray, ...]:
    root = Path(base_dir) if base_dir else DATA_DIR
    target_dir = root / market

    if not target_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {target_dir}")

    dates = np.load(target_dir / "dates.npy", allow_pickle=True)
    symbols = np.load(target_dir / "symbols.npy", allow_pickle=True)

    market_features = np.load(target_dir / "market_features.npy", mmap_mode="r", allow_pickle=False)
    asset_features = np.load(target_dir / "asset_features.npy", mmap_mode="r", allow_pickle=False)
    ror_array = np.load(target_dir / "ror_array.npy", mmap_mode="r", allow_pickle=False)

    meta_path = target_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        shapes = meta.get("shapes") or {}
        meta_labels = ["dates", "symbols", "market_features", "asset_features", "ror_array"]
        arrays = [dates, symbols, market_features, asset_features, ror_array]

        for name, arr in zip(meta_labels, arrays):
            exp = shapes.get(name)
            if exp is not None:
                if tuple(exp) != tuple(arr.shape):
                    raise ValueError(
                        f"[{market}] Shape mismatch for '{name}':\n"
                        f"  Expected (Meta): {tuple(exp)}\n"
                        f"  Actual (Numpy):  {arr.shape}"
                    )

    return dates, market_features, symbols, asset_features, ror_array
