"""Backwards-compatible shim.

The real implementations live in `src/olist/`. This file exists so any older
notebook that did `from utils import load_data, save_parquet, check_data` keeps
working. New code should import from `olist` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "olist").exists())
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from olist import config as cfg
from olist.io_utils import load_parquet_dir, load_raw
from olist.io_utils import save_parquet as _save_parquet


def load_data(path: str | Path | None = None) -> dict:
    """Load a directory of CSV/parquet tables. Cross-platform, unlike the old
    hard-coded Windows path."""
    path = Path(path) if path is not None else cfg.RAW_DIR
    if any(path.glob("*.parquet")):
        return load_parquet_dir(path)
    return load_raw(path)


def save_parquet(save_path, df, dataset_name):
    """Old argument order kept on purpose."""
    return _save_parquet(df, save_path, dataset_name)


def check_data(df: pd.DataFrame) -> None:
    print("Columns:", list(df.columns))
    print(df.dtypes)
    print(df.describe())
    df.info()
    print("*" * 80)
