"""Reading and writing tables. Raw files are opened read-only, never rewritten."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from . import config as cfg


class MissingRawDataError(FileNotFoundError):
    """Raised with actionable instructions instead of a bare traceback."""


def load_raw(raw_dir: Path | str | None = None) -> Dict[str, pd.DataFrame]:
    """Load the nine Olist CSVs into a dict of DataFrames.

    Raw files are read only. Nothing in this project writes to data/raw.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else cfg.RAW_DIR
    if not raw_dir.exists():
        raise MissingRawDataError(
            f"{raw_dir} does not exist.\n"
            "Download the Olist dataset and unzip the nine CSVs into data/raw/.\n"
            f"Source: {cfg.DATASET_SOURCE}\n"
            "For a smoke test without the real data run: python scripts/make_sample_data.py"
        )

    tables: Dict[str, pd.DataFrame] = {}
    missing = []
    for name, filename in cfg.RAW_TABLES.items():
        path = raw_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        tables[name] = pd.read_csv(path)

    if missing:
        raise MissingRawDataError(
            f"Missing {len(missing)} expected file(s) in {raw_dir}: {', '.join(missing)}\n"
            f"Source: {cfg.DATASET_SOURCE}"
        )
    return tables


def load_parquet_dir(directory: Path | str) -> Dict[str, pd.DataFrame]:
    """Load every .parquet file in a directory, keyed by filename stem."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} is empty. Run `python scripts/run_pipeline.py` first."
        )
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No parquet files in {directory}. Run `python scripts/run_pipeline.py` first."
        )
    return {p.stem: pd.read_parquet(p) for p in paths}


def read_table(directory: Path, table: str) -> pd.DataFrame:
    path = Path(directory) / f"{table}.parquet"
    if not path.exists():
        available = sorted(p.stem for p in Path(directory).glob("*.parquet"))
        raise FileNotFoundError(
            f"{path} not found. Available in {directory}: {available or 'nothing'}.\n"
            "Run `python scripts/run_pipeline.py` to build it."
        )
    return pd.read_parquet(path)


def load_clean(table: str) -> pd.DataFrame:
    return read_table(cfg.CLEAN_DIR, table)


def load_features(table: str) -> pd.DataFrame:
    return read_table(cfg.FEATURES_DIR, table)


def load_ml(table: str) -> pd.DataFrame:
    return read_table(cfg.ML_DIR, table)


def save_parquet(df: pd.DataFrame, directory: Path | str, name: str) -> Path:
    """Write a DataFrame to parquet, normalising dtypes parquet cannot hold.

    `category` and `period` columns are cast to string so the file stays
    readable by pandas, DuckDB and Power BI without extra configuration.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if str(dtype) == "category" or str(dtype).startswith("period"):
            out[col] = out[col].astype(str)
        elif dtype == object:
            # object columns holding datetime.date break some parquet readers
            non_null = out[col].dropna()
            if len(non_null) and hasattr(non_null.iloc[0], "isoformat"):
                out[col] = out[col].astype(str)
    path = directory / f"{name}.parquet"
    out.to_parquet(path, index=False)
    return path


def save_json(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def table_summary(tables: Dict[str, pd.DataFrame], expected: dict | None = None) -> pd.DataFrame:
    """One row per table: shape, memory, null share, duplicate share."""
    rows = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": df.shape[1],
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
                "null_cells_pct": round(df.isna().to_numpy().mean() * 100, 2),
                "duplicate_rows": int(df.duplicated().sum()),
                "expected_rows": (expected or {}).get(name),
            }
        )
    summary = pd.DataFrame(rows)
    if expected:
        summary["rows_match"] = summary["rows"].eq(summary["expected_rows"])
    return summary.sort_values("table").reset_index(drop=True)
