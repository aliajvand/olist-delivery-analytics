"""Run Phase 1 (clean) and Phase 2 (features) headlessly.

The notebooks tell the story; this script is what you run to rebuild the data.
Reproducing the project should never require opening Jupyter.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --raw-dir data/sample_raw --out-dir data/sample_processed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from olist import cleaning, config as cfg, features, io_utils as io  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=None, help="defaults to data/raw")
    parser.add_argument("--out-dir", default=None, help="defaults to data/processed")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.RAW_DIR
    out_dir = Path(args.out_dir) if args.out_dir else cfg.PROCESSED_DIR
    clean_dir, features_dir, ml_dir = out_dir / "cleaning", out_dir / "features", out_dir / "ml"

    pd.set_option("display.width", 200)
    log = (lambda *a: None) if args.quiet else print

    t0 = time.time()
    log(f"[1/4] loading raw CSVs from {raw_dir}")
    raw = io.load_raw(raw_dir)
    log(io.table_summary(raw, cfg.EXPECTED_RAW_ROWS).to_string(index=False), "\n")

    log("[2/4] cleaning")
    clean, clean_audits = cleaning.build_clean_tables(raw)
    for name, audit in clean_audits.items():
        log(f"-- {name}")
        log(audit.to_string(index=False), "\n")
    for name, df in clean.items():
        io.save_parquet(df, clean_dir, name)
    log(f"     wrote {len(clean)} tables to {clean_dir}\n")

    log("[3/4] features")
    feature_tables, feature_audits = features.build_feature_tables(clean)
    for name, audit in feature_audits.items():
        log(f"-- {name}")
        log(audit.to_string(index=False), "\n")
    for name, df in feature_tables.items():
        target_dir = ml_dir if name == "order_model_base" else features_dir
        io.save_parquet(df, target_dir, name)
        log(f"     {name:<32} {df.shape}")

    log("\n[4/4] gate checks")
    grain_failed = feature_audits["grain"].query("status != 'PASS'")
    recon_failed = feature_audits["reconciliation"].query("status != 'PASS'")
    if len(grain_failed) or len(recon_failed):
        log("FAILED. Fix these before modelling:")
        log(pd.concat([grain_failed, recon_failed]).to_string(index=False))
        return 1

    base = feature_tables["order_model_base"]
    log(f"     grain + reconciliation: PASS")
    log(f"     order_model_base: {base.shape}, splits = {base['split'].value_counts().to_dict()}")
    log(f"     late rate (delivered orders): {base['late_delivery_flag'].mean():.4f}")
    log(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
