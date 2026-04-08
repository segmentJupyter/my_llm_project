"""
Shared paths + loading processed datasets (used by Flask app and CLI training).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from utils.parsing import parse_dataset_file

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"

ALLOWED_EXT = {".csv", ".txt"}


def safe_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    return name[:180] if name else "dataset"


def dataset_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return safe_name(base)


def processed_path(raw_name: str) -> Path:
    return DATA_DIR / f"{dataset_id(raw_name)}.processed.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df


def load_processed_dataset(raw_name: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns df and meta. If processed CSV doesn't exist, parse raw file and create it.
    """
    raw_path = DATA_DIR / raw_name
    proc_path = processed_path(raw_name)

    if proc_path.exists():
        df = pd.read_csv(proc_path)
        meta = {"timestamp_col": "timestamp", "energy_col": "energy"}
        return df, meta

    parsed = parse_dataset_file(str(raw_path))
    df = parsed.df.copy()
    df = df.rename(columns={parsed.timestamp_col: "timestamp", parsed.energy_col: "energy"})

    lower_to_orig = {c.lower(): c for c in df.columns}
    for key, aliases in {
        "temperature": ("temperature", "temp"),
        "humidity": ("humidity",),
        "price": ("price",),
        "demand": ("demand",),
    }.items():
        if key in df.columns:
            continue
        for a in aliases:
            for lc, orig in lower_to_orig.items():
                if a == lc or a in lc:
                    if orig != key and orig in df.columns:
                        df = df.rename(columns={orig: key})
                    break

    df = _normalize_columns(df)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_path, index=False)
    meta = {"timestamp_col": "timestamp", "energy_col": "energy"}
    return df, meta


def list_raw_datasets() -> List[str]:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    files = []
    for p in DATA_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            files.append(p.name)
    return sorted(files)


def trained_dataset_ids() -> Set[str]:
    """Dataset IDs that have saved LSTM metadata (files named `<id>.meta.json`)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out: Set[str] = set()
    for p in ARTIFACTS_DIR.glob("*.meta.json"):
        name = p.name
        if name.endswith(".meta.json"):
            out.add(name[: -len(".meta.json")])
    return out


def has_trained_model(raw_filename: str) -> bool:
    did = dataset_id(raw_filename)
    meta = ARTIFACTS_DIR / f"{did}.meta.json"
    pt = ARTIFACTS_DIR / f"{did}.pt"
    return meta.exists() and pt.exists()


def read_training_meta(raw_filename: str) -> Optional[Dict]:
    did = dataset_id(raw_filename)
    path = ARTIFACTS_DIR / f"{did}.meta.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
