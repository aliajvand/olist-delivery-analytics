"""Olist delivery-performance analytics.

Layout:
    config    - paths, cutoff dates, feature contract constants
    io_utils  - reading raw CSVs, reading/writing parquet
    cleaning  - Phase 1, raw -> clean tables (pure functions)
    features  - Phase 2, clean -> feature tables + ML base table
    modeling  - Phase 4, split, baselines, candidates, policies, artifacts
    plots     - figure helpers shared by the EDA and modelling notebooks
"""
from . import cleaning, config, features, io_utils, modeling, plots  # noqa: F401

__all__ = ["config", "io_utils", "cleaning", "features", "modeling", "plots"]
__version__ = "1.0.0"
