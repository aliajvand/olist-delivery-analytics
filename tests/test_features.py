"""The features that matter are the leakage-safe ones, so those get the most tests."""
import numpy as np
import pandas as pd

from olist import config as cfg, features


def test_declared_grain_holds(feature_tables):
    audit = features.grain_audit(feature_tables)
    assert (audit["status"] == "PASS").all(), audit.to_string(index=False)


def test_gmv_reconciles_to_item_prices(clean, feature_tables):
    recon = features.feature_reconciliation(clean, feature_tables)
    assert (recon["status"] == "PASS").all(), recon.to_string(index=False)


def test_late_flag_is_nan_for_undelivered_orders(feature_tables):
    of = feature_tables["order_features"]
    undelivered = of["Order_Delivered_Customer_Date"].isna()
    assert of.loc[undelivered, "late_delivery_flag"].isna().all(), (
        "an undelivered order must be NaN, not 0 - filling it with 0 invents on-time deliveries"
    )
    assert of.loc[~undelivered, "late_delivery_flag"].notna().all()


def test_late_flag_matches_its_definition(feature_tables):
    of = feature_tables["order_features"].dropna(subset=["delivery_delay_days"])
    expected = (of["delivery_delay_days"] > cfg.LATE_TOLERANCE_DAYS).astype(float)
    assert np.allclose(of["late_delivery_flag"], expected)


def test_haversine_against_known_distance():
    # Sao Paulo -> Rio de Janeiro is about 360 km
    km = features.haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 330 < km < 390


def test_model_base_contains_no_post_outcome_predictors():
    banned = set(cfg.POST_OUTCOME_COLUMNS)
    leaked = banned.intersection(features.PURCHASE_TIME_FEATURES)
    assert not leaked, f"feature contract includes post-outcome columns: {leaked}"


def test_splits_are_chronological_and_disjoint(feature_tables):
    base = feature_tables["order_model_base"]
    bounds = base.groupby("split")["Order_Purchase_Timestamp"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["valid", "min"]
    assert bounds.loc["valid", "max"] < bounds.loc["test", "min"]


def test_seller_prior_rate_never_sees_the_future(feature_tables, clean):
    """The core leakage guarantee.

    A seller's first-ever order must carry zero prior observations. If the naive
    groupby-mean had been used, prior_orders would be the seller's full lifetime
    count instead.
    """
    of = feature_tables["order_features"]
    items = clean["olist_order_items_dataset"]
    hist = features.as_of_reliability(of, items, key="Seller_Id", prefix="seller")

    link = items[["Order_Id", "Seller_Id"]].drop_duplicates()
    first_orders = (
        of[["Order_Id", "Order_Purchase_Timestamp"]]
        .merge(link, on="Order_Id")
        .sort_values("Order_Purchase_Timestamp")
        .drop_duplicates(subset="Seller_Id", keep="first")["Order_Id"]
    )
    firsts = hist[hist["Order_Id"].isin(first_orders)]
    assert (firsts["seller_prior_orders"] == 0).all()


def test_seller_prior_orders_is_monotonic_per_seller(feature_tables, clean):
    of = feature_tables["order_features"]
    items = clean["olist_order_items_dataset"]
    hist = features.as_of_reliability(of, items, key="Seller_Id", prefix="seller")
    single = items.groupby("Order_Id")["Seller_Id"].nunique().eq(1)
    single_orders = single[single].index

    merged = (
        hist[hist["Order_Id"].isin(single_orders)]
        .merge(of[["Order_Id", "Order_Purchase_Timestamp"]], on="Order_Id")
        .merge(items[["Order_Id", "Seller_Id"]].drop_duplicates(), on="Order_Id")
        .sort_values(["Seller_Id", "Order_Purchase_Timestamp"])
    )
    diffs = merged.groupby("Seller_Id")["seller_prior_orders"].diff().dropna()
    assert (diffs >= 0).all(), "prior order count went down over time"


def test_smoothed_rate_stays_in_range(feature_tables, clean):
    of = feature_tables["order_features"]
    items = clean["olist_order_items_dataset"]
    hist = features.as_of_reliability(of, items, key="Seller_Id", prefix="seller")
    rate = hist["seller_prior_late_rate"]
    assert rate.between(0, 1).all()
    assert rate.notna().all()


def test_customer_history_excludes_the_current_order(feature_tables):
    of = feature_tables["order_features"]
    hist = features.as_of_customer_history(of)
    merged = of[["Order_Id", "Customer_Unique_Id", "Order_Purchase_Timestamp"]].merge(
        hist, on="Order_Id"
    )
    firsts = (
        merged.sort_values("Order_Purchase_Timestamp")
        .drop_duplicates(subset="Customer_Unique_Id", keep="first")
    )
    assert (firsts["customer_prior_orders"] == 0).all()
    assert (firsts["customer_prior_revenue"] == 0).all()


def test_rfm_scores_are_in_range(feature_tables):
    rfm = feature_tables["customer_features"]
    for col in ("r_score", "f_score", "m_score"):
        assert rfm[col].between(1, 5).all()
    assert rfm["recency_days"].ge(0).all()


def test_estimated_delivery_days_is_known_at_purchase(feature_tables):
    of = feature_tables["order_features"]
    assert of["estimated_delivery_days"].notna().mean() > 0.99
    assert of["estimated_delivery_days"].gt(0).all()
