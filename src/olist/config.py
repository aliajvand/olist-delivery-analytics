"""Single source of truth for paths, constants and modelling conventions.

Everything else in the project imports from here. If a path or a cutoff date
needs to change, it changes in exactly one place.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is resolved from this file, so notebooks, scripts and tests all
# agree on it regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("OLIST_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = PROCESSED_DIR / "cleaning"
FEATURES_DIR = PROCESSED_DIR / "features"
ML_DIR = PROCESSED_DIR / "ml"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_FIGURES_DIR = FIGURES_DIR / "model"
MODELS_DIR = PROJECT_ROOT / "models"

ALL_OUTPUT_DIRS = (
    CLEAN_DIR,
    FEATURES_DIR,
    ML_DIR,
    FIGURES_DIR,
    MODEL_FIGURES_DIR,
    MODELS_DIR,
)


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for directory in ALL_OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset contract
# --------------------------------------------------------------------------
RAW_TABLES = {
    "olist_customers_dataset": "olist_customers_dataset.csv",
    "olist_geolocation_dataset": "olist_geolocation_dataset.csv",
    "olist_order_items_dataset": "olist_order_items_dataset.csv",
    "olist_order_payments_dataset": "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset": "olist_order_reviews_dataset.csv",
    "olist_orders_dataset": "olist_orders_dataset.csv",
    "olist_products_dataset": "olist_products_dataset.csv",
    "olist_sellers_dataset": "olist_sellers_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv",
}

# Row counts of the canonical Kaggle release. Used as a soft integrity check in
# notebook 01 so a truncated or re-sampled download is caught immediately.
EXPECTED_RAW_ROWS = {
    "olist_customers_dataset": 99441,
    "olist_geolocation_dataset": 1000163,
    "olist_order_items_dataset": 112650,
    "olist_order_payments_dataset": 103886,
    "olist_order_reviews_dataset": 99224,
    "olist_orders_dataset": 99441,
    "olist_products_dataset": 32951,
    "olist_sellers_dataset": 3095,
    "product_category_name_translation": 71,
}

DATASET_SOURCE = (
    "Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0), "
    "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
)

# --------------------------------------------------------------------------
# Modelling conventions
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5

# The data ends here. Never use pd.Timestamp.now() for recency or as-of logic:
# it would silently change every metric depending on when the code is run.
SNAPSHOT_DATE = pd.Timestamp("2018-10-18")

# Chronological holdout. Orders are split by purchase timestamp so the
# validation and test periods sit strictly after the training period, which is
# how the model would actually be used.
TRAIN_END = pd.Timestamp("2018-05-01")
VALID_END = pd.Timestamp("2018-07-01")
# test = purchase_ts >= VALID_END

# "Late" means the customer received the order after the promised date.
LATE_TOLERANCE_DAYS = 0.0

# Columns that must never enter a feature matrix: they are either identifiers
# or only knowable after the outcome we are trying to predict.
IDENTIFIER_PATTERNS = ("_id",)
POST_OUTCOME_COLUMNS = (
    "Order_Approved_At",
    "Order_Delivered_Carrier_Date",
    "Order_Delivered_Customer_Date",
    "Order_Status",
    "approval_hours",
    "carrier_hours",
    "last_mile_hours",
    "total_delivery_hours",
    "delivery_days",
    "delivery_delay_days",
    "late_delivery_flag",
    "cancelled_flag",
    "Is_Approved",
    "Is_Shipped",
    "Is_Delivered",
    "review_score",
)


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed the two RNGs scikit-learn actually reads."""
    random.seed(seed)
    np.random.seed(seed)
