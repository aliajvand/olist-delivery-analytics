"""Phase 2 - Features.

Grain declarations (the thing reviewers ask about first):
  order_features    -> one row per Order_Id
  customer_features -> one row per Customer_Unique_Id, as of SNAPSHOT_DATE
  product_features  -> one row per Product_Id
  seller_features   -> one row per Seller_Id
  review_features   -> one row per Order_Id (reviews were deduplicated upstream)
  order_geo         -> one row per Order_Id
  order_model_base  -> one row per Order_Id, ML-ready, leakage audited

Observation window and horizon:
  Decision point = Order_Purchase_Timestamp. Every predictor in
  order_model_base is knowable at that instant. Every target is realised after
  it. Historical aggregates (seller reliability, route reliability) are built
  with an as-of join on *delivery* time, so an order can only see outcomes that
  had already been observed when it was placed.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as cfg

EARTH_RADIUS_KM = 6371.0


def _to_ns(series: pd.Series) -> pd.Series:
    """merge_asof requires both join keys at identical datetime resolution.

    pandas 2 and pandas 3 infer different units from the same CSV, so pin it.
    """
    return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")


# --------------------------------------------------------------------------
# Geo
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorised over pandas Series or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_order_geo(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """One row per order: customer/seller coordinates and shipping distance.

    A multi-seller order has several distances. We keep max (the leg that sets
    the delivery date), mean, and the seller count, instead of picking one
    arbitrary row.
    """
    geo = geolocation[["Geolocation_Zip_Code_Prefix", "Geolocation_Lat", "Geolocation_Lng"]]

    customers_geo = customers.merge(
        geo, left_on="Customer_Zip_Code_Prefix", right_on="Geolocation_Zip_Code_Prefix", how="left"
    ).rename(columns={"Geolocation_Lat": "Customer_Lat", "Geolocation_Lng": "Customer_Lng"})

    sellers_geo = sellers.merge(
        geo, left_on="Seller_Zip_Code_Prefix", right_on="Geolocation_Zip_Code_Prefix", how="left"
    ).rename(columns={"Geolocation_Lat": "Seller_Lat", "Geolocation_Lng": "Seller_Lng"})

    legs = (
        items[["Order_Id", "Seller_Id"]]
        .drop_duplicates()
        .merge(orders[["Order_Id", "Customer_Id"]], on="Order_Id", how="left")
        .merge(
            customers_geo[["Customer_Id", "Customer_Lat", "Customer_Lng", "Customer_State", "Customer_Zip_Code_Prefix"]],
            on="Customer_Id",
            how="left",
        )
        .merge(
            sellers_geo[["Seller_Id", "Seller_Lat", "Seller_Lng", "Seller_State", "Seller_Zip_Code_Prefix"]],
            on="Seller_Id",
            how="left",
        )
    )
    legs["leg_distance_km"] = haversine_km(
        legs["Customer_Lat"], legs["Customer_Lng"], legs["Seller_Lat"], legs["Seller_Lng"]
    )
    legs["leg_same_state"] = (
        legs["Customer_State"].astype("string") == legs["Seller_State"].astype("string")
    ).astype("int8")

    def dominant(series: pd.Series):
        mode = series.dropna().mode()
        return mode.iloc[0] if len(mode) else pd.NA

    order_geo = (
        legs.groupby("Order_Id", observed=True)
        .agg(
            distance_km=("leg_distance_km", "max"),
            distance_km_mean=("leg_distance_km", "mean"),
            n_shipping_legs=("Seller_Id", "nunique"),
            same_state_flag=("leg_same_state", "min"),
            Customer_State=("Customer_State", dominant),
            Seller_State=("Seller_State", dominant),
            Customer_Lat=("Customer_Lat", "first"),
            Customer_Lng=("Customer_Lng", "first"),
            Seller_Lat=("Seller_Lat", "mean"),
            Seller_Lng=("Seller_Lng", "mean"),
        )
        .reset_index()
    )
    order_geo["distance_bucket"] = pd.cut(
        order_geo["distance_km"],
        bins=[-0.01, 50, 200, 500, 1000, np.inf],
        labels=["<50km", "50-200km", "200-500km", "500-1000km", "1000km+"],
    ).astype("string")
    return order_geo


# --------------------------------------------------------------------------
# Order features
# --------------------------------------------------------------------------
def build_order_features(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    order_geo: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """One row per order: basket, payment, timing, physical and geo attributes."""
    item_products = items.merge(
        products[
            [
                "Product_Id",
                "Product_Category_Name_English",
                "Product_Weight_G",
                "Product_Volume_Cm3",
                "Product_Photos_Qty",
            ]
        ],
        on="Product_Id",
        how="left",
    )

    def dominant(series: pd.Series):
        mode = series.dropna().mode()
        return mode.iloc[0] if len(mode) else pd.NA

    item_agg = (
        item_products.groupby("Order_Id", observed=True)
        .agg(
            order_item_count=("Order_Item_Id", "count"),
            order_product_count=("Product_Id", "nunique"),
            order_seller_count=("Seller_Id", "nunique"),
            order_gmv=("Price", "sum"),
            order_freight_total=("Freight_Value", "sum"),
            max_item_price=("Price", "max"),
            order_weight_g=("Product_Weight_G", "sum"),
            order_volume_cm3=("Product_Volume_Cm3", "sum"),
            max_item_weight_g=("Product_Weight_G", "max"),
            main_category=("Product_Category_Name_English", dominant),
            shipping_limit_date=("Shipping_Limit_Date", "max"),
        )
        .reset_index()
    )

    pay_agg = (
        payments.groupby("Order_Id", observed=True)
        .agg(
            order_total_paid=("Payment_Value", "sum"),
            payment_count=("Payment_Sequential", "count"),
            max_installments=("Payment_Installments", "max"),
            payment_method_count=("Payment_Type", "nunique"),
        )
        .reset_index()
    )
    dominant_payment = (
        payments.sort_values("Payment_Value", ascending=False)
        .drop_duplicates(subset="Order_Id", keep="first")[["Order_Id", "Payment_Type"]]
        .rename(columns={"Payment_Type": "dominant_payment_type"})
    )

    of = (
        orders.merge(item_agg, on="Order_Id", how="left")
        .merge(pay_agg, on="Order_Id", how="left")
        .merge(dominant_payment, on="Order_Id", how="left")
        .merge(customers[["Customer_Id", "Customer_Unique_Id"]], on="Customer_Id", how="left")
        .merge(order_geo.drop(columns=["Customer_Lat", "Customer_Lng"]), on="Order_Id", how="left")
    )

    # ---- known at purchase time -----------------------------------------
    purchase = of["Order_Purchase_Timestamp"]
    of["purchase_date"] = purchase.dt.normalize()
    of["purchase_month"] = purchase.dt.to_period("M").astype(str)
    of["purchase_year"] = purchase.dt.year.astype("Int16")
    of["purchase_weekday"] = purchase.dt.day_name().astype("string")
    of["purchase_weekday_num"] = purchase.dt.weekday.astype("Int8")
    of["purchase_hour"] = purchase.dt.hour.astype("Int8")
    of["purchase_is_weekend"] = purchase.dt.weekday.ge(5).astype("int8")
    of["days_since_dataset_start"] = (purchase - purchase.min()).dt.days.astype("Int32")

    of["estimated_delivery_days"] = (
        (of["Order_Estimated_Delivery_Date"] - purchase).dt.total_seconds() / 86400
    )
    of["shipping_limit_days"] = (
        (of["shipping_limit_date"] - purchase).dt.total_seconds() / 86400
    )
    of["freight_ratio"] = np.where(
        of["order_gmv"] > 0, of["order_freight_total"] / of["order_gmv"], np.nan
    )
    of["avg_item_price"] = np.where(
        of["order_item_count"].astype("Float64").fillna(0) > 0,
        of["order_gmv"] / of["order_item_count"].astype("Float64"),
        np.nan,
    )
    of["freight_per_kg"] = np.where(
        of["order_weight_g"] > 0, of["order_freight_total"] / (of["order_weight_g"] / 1000), np.nan
    )
    of["is_multi_seller"] = of["order_seller_count"].fillna(0).gt(1).astype("int8")

    # ---- realised after purchase: targets and KPIs only ------------------
    of["approval_hours"] = (of["Order_Approved_At"] - purchase).dt.total_seconds() / 3600
    of["carrier_hours"] = (
        of["Order_Delivered_Carrier_Date"] - of["Order_Approved_At"]
    ).dt.total_seconds() / 3600
    of["last_mile_hours"] = (
        of["Order_Delivered_Customer_Date"] - of["Order_Delivered_Carrier_Date"]
    ).dt.total_seconds() / 3600
    of["total_delivery_hours"] = (
        of["Order_Delivered_Customer_Date"] - purchase
    ).dt.total_seconds() / 3600
    of["delivery_days"] = of["total_delivery_hours"] / 24

    of["delivery_delay_days"] = (
        of["Order_Delivered_Customer_Date"] - of["Order_Estimated_Delivery_Date"]
    ).dt.total_seconds() / 86400

    # NaN, not 0, for undelivered orders. A missing outcome is not a good outcome.
    of["late_delivery_flag"] = np.where(
        of["Order_Delivered_Customer_Date"].notna(),
        (of["delivery_delay_days"] > cfg.LATE_TOLERANCE_DAYS).astype(float),
        np.nan,
    )
    of["cancelled_flag"] = of["Order_Status"].astype("string").eq("canceled").astype("int8")
    of["delivered_flag"] = of["Order_Delivered_Customer_Date"].notna().astype("int8")
    return of


# --------------------------------------------------------------------------
# Leakage-safe historical aggregates
# --------------------------------------------------------------------------
def as_of_reliability(
    order_features: pd.DataFrame,
    items: pd.DataFrame,
    key: str = "Seller_Id",
    prefix: str = "seller",
    prior_weight: float = 20.0,
) -> pd.DataFrame:
    """Historical late rate per key, as known at each order's purchase time.

    The naive version, `groupby(key)['late_delivery_flag'].mean()`, leaks: it
    includes outcomes from the future relative to the order being scored.

    This version:
      1. builds an event log of (key, delivery_time, was_late) for delivered orders
      2. accumulates count and late-count per key, ordered by delivery time
      3. as-of joins each target order on its *purchase* time, so it can only
         see deliveries that had already completed
      4. smooths towards the global rate with `prior_weight` pseudo-observations,
         so a seller with two shipments does not get a 0% or 100% reputation

    Returns one row per Order_Id with `<prefix>_prior_orders` and `<prefix>_prior_late_rate`.
    """
    order_time = order_features[["Order_Id", "Order_Purchase_Timestamp"]].copy()
    order_time["Order_Purchase_Timestamp"] = _to_ns(order_time["Order_Purchase_Timestamp"])
    link = items[["Order_Id", key]].drop_duplicates()

    events = (
        order_features.loc[
            order_features["Order_Delivered_Customer_Date"].notna(),
            ["Order_Id", "Order_Delivered_Customer_Date", "late_delivery_flag"],
        ]
        .merge(link, on="Order_Id", how="inner")
        .dropna(subset=[key])
        .rename(columns={"Order_Delivered_Customer_Date": "event_time"})
        .assign(event_time=lambda d: _to_ns(d["event_time"]))
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    events["cum_n"] = events.groupby(key, observed=True).cumcount() + 1
    events["cum_late"] = events.groupby(key, observed=True)["late_delivery_flag"].cumsum()

    global_rate = float(order_features["late_delivery_flag"].mean(skipna=True))

    targets = (
        link.merge(order_time, on="Order_Id", how="inner")
        .dropna(subset=[key, "Order_Purchase_Timestamp"])
        .sort_values("Order_Purchase_Timestamp")
        .reset_index(drop=True)
    )

    joined = pd.merge_asof(
        targets,
        events[[key, "event_time", "cum_n", "cum_late"]],
        left_on="Order_Purchase_Timestamp",
        right_on="event_time",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    )
    joined["cum_n"] = joined["cum_n"].fillna(0)
    joined["cum_late"] = joined["cum_late"].fillna(0)

    # Multi-seller orders: average the reputations of the sellers involved.
    per_order = (
        joined.groupby("Order_Id", observed=True)
        .agg(prior_orders=("cum_n", "sum"), prior_late=("cum_late", "sum"))
        .reset_index()
    )
    per_order["prior_late_rate"] = (per_order["prior_late"] + prior_weight * global_rate) / (
        per_order["prior_orders"] + prior_weight
    )
    per_order = per_order.rename(
        columns={
            "prior_orders": f"{prefix}_prior_orders",
            "prior_late_rate": f"{prefix}_prior_late_rate",
        }
    ).drop(columns=["prior_late"])
    return per_order


def as_of_route_reliability(order_features: pd.DataFrame, prior_weight: float = 30.0) -> pd.DataFrame:
    """Same as-of logic, keyed on the customer state -> seller state route."""
    df = order_features[
        [
            "Order_Id",
            "Order_Purchase_Timestamp",
            "Order_Delivered_Customer_Date",
            "late_delivery_flag",
            "Customer_State",
            "Seller_State",
        ]
    ].copy()
    df["Order_Purchase_Timestamp"] = _to_ns(df["Order_Purchase_Timestamp"])
    df["Order_Delivered_Customer_Date"] = _to_ns(df["Order_Delivered_Customer_Date"])
    df["route"] = (
        df["Seller_State"].astype("string").fillna("NA")
        + "->"
        + df["Customer_State"].astype("string").fillna("NA")
    )

    events = (
        df.loc[df["Order_Delivered_Customer_Date"].notna()]
        .rename(columns={"Order_Delivered_Customer_Date": "event_time"})
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    events["cum_n"] = events.groupby("route", observed=True).cumcount() + 1
    events["cum_late"] = events.groupby("route", observed=True)["late_delivery_flag"].cumsum()

    global_rate = float(df["late_delivery_flag"].mean(skipna=True))

    targets = df[["Order_Id", "Order_Purchase_Timestamp", "route"]].sort_values(
        "Order_Purchase_Timestamp"
    ).reset_index(drop=True)

    joined = pd.merge_asof(
        targets,
        events[["route", "event_time", "cum_n", "cum_late"]],
        left_on="Order_Purchase_Timestamp",
        right_on="event_time",
        by="route",
        direction="backward",
        allow_exact_matches=False,
    )
    joined["cum_n"] = joined["cum_n"].fillna(0)
    joined["cum_late"] = joined["cum_late"].fillna(0)
    joined["route_prior_late_rate"] = (joined["cum_late"] + prior_weight * global_rate) / (
        joined["cum_n"] + prior_weight
    )
    return joined[["Order_Id", "route", "cum_n", "route_prior_late_rate"]].rename(
        columns={"cum_n": "route_prior_orders"}
    )


def as_of_customer_history(order_features: pd.DataFrame) -> pd.DataFrame:
    """Per order: how many orders and how much revenue this customer had *before* it."""
    df = (
        order_features[["Order_Id", "Customer_Unique_Id", "Order_Purchase_Timestamp", "order_total_paid"]]
        .dropna(subset=["Customer_Unique_Id"])
        .sort_values(["Customer_Unique_Id", "Order_Purchase_Timestamp"])
        .reset_index(drop=True)
    )
    grouped = df.groupby("Customer_Unique_Id", observed=True)
    df["prior_orders"] = grouped.cumcount()
    paid = df["order_total_paid"].fillna(0)
    df["prior_revenue"] = (
        paid.groupby(df["Customer_Unique_Id"]).cumsum() - paid
    )
    df["is_repeat_purchase"] = df["prior_orders"].gt(0).astype("int8")
    return df[["Order_Id", "prior_orders", "prior_revenue", "is_repeat_purchase"]].rename(
        columns={"prior_orders": "customer_prior_orders", "prior_revenue": "customer_prior_revenue"}
    )


# --------------------------------------------------------------------------
# Customer / product / seller / review features
# --------------------------------------------------------------------------
def build_customer_features(
    order_features: pd.DataFrame, snapshot_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """RFM at Customer_Unique_Id grain, as of a fixed documented snapshot."""
    snapshot_date = snapshot_date or cfg.SNAPSHOT_DATE
    base = order_features[
        (order_features["Order_Status"].astype("string") == "delivered")
        & (order_features["Order_Purchase_Timestamp"] <= snapshot_date)
    ].copy()

    rfm = (
        base.groupby("Customer_Unique_Id", observed=True)
        .agg(
            last_purchase=("Order_Purchase_Timestamp", "max"),
            first_purchase=("Order_Purchase_Timestamp", "min"),
            frequency_orders=("Order_Id", "nunique"),
            monetary_value=("order_total_paid", "sum"),
            avg_delivery_days=("delivery_days", "mean"),
            late_orders=("late_delivery_flag", "sum"),
            total_items=("order_item_count", "sum"),
            main_state=("Customer_State", "first"),
        )
        .reset_index()
    )
    rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days
    rfm["tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
    rfm["repeat_customer_flag"] = rfm["frequency_orders"].gt(1).astype("int8")
    rfm["aov"] = rfm["monetary_value"] / rfm["frequency_orders"]
    rfm["late_rate"] = rfm["late_orders"] / rfm["frequency_orders"]

    # Frequency is 1 for ~97% of customers, so qcut on the raw value cannot form
    # 5 bins. Rank-then-qcut with duplicates="drop" keeps this honest.
    rfm["r_score"] = _score(rfm["recency_days"], ascending=False)
    rfm["f_score"] = _score(rfm["frequency_orders"].rank(method="first"), ascending=True)
    rfm["m_score"] = _score(rfm["monetary_value"], ascending=True)
    rfm["rfm_segment"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )
    rfm["rfm_segment_label"] = [
        _segment_label(r, f, m)
        for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"])
    ]
    return rfm


def _score(series: pd.Series, ascending: bool, bins: int = 5) -> pd.Series:
    labels = list(range(1, bins + 1)) if ascending else list(range(bins, 0, -1))
    try:
        return pd.qcut(series, bins, labels=labels).astype(int)
    except ValueError:
        ranked = series.rank(method="first", pct=True)
        edges = np.linspace(0, 1, bins + 1)
        return pd.cut(ranked, bins=edges, labels=labels, include_lowest=True).astype(int)


def _segment_label(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 4 and m >= 4:
        return "Big Spenders"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r <= 2 and (f >= 3 or m >= 4):
        return "At Risk"
    if r <= 2:
        return "Lost"
    return "Others"


def build_product_features(items: pd.DataFrame, products: pd.DataFrame, order_features: pd.DataFrame) -> pd.DataFrame:
    item_order = items.merge(
        order_features[["Order_Id", "late_delivery_flag", "cancelled_flag", "delivered_flag"]],
        on="Order_Id",
        how="left",
    )
    pf = (
        item_order.groupby("Product_Id", observed=True)
        .agg(
            product_sales_count=("Order_Item_Id", "count"),
            product_order_count=("Order_Id", "nunique"),
            product_revenue=("Price", "sum"),
            average_price=("Price", "mean"),
            average_freight=("Freight_Value", "mean"),
            product_late_rate=("late_delivery_flag", "mean"),
            product_cancel_rate=("cancelled_flag", "mean"),
        )
        .reset_index()
    )
    pf["freight_to_price_ratio"] = np.where(
        pf["average_price"] > 0, pf["average_freight"] / pf["average_price"], np.nan
    )
    return pf.merge(
        products[
            [
                "Product_Id",
                "Product_Category_Name",
                "Product_Category_Name_English",
                "Product_Weight_G",
                "Product_Volume_Cm3",
            ]
        ],
        on="Product_Id",
        how="left",
    )


def build_seller_features(items: pd.DataFrame, sellers: pd.DataFrame, order_features: pd.DataFrame) -> pd.DataFrame:
    item_order = items.merge(
        order_features[
            ["Order_Id", "late_delivery_flag", "cancelled_flag", "delivery_days", "delivery_delay_days"]
        ],
        on="Order_Id",
        how="left",
    )
    sf = (
        item_order.groupby("Seller_Id", observed=True)
        .agg(
            seller_order_count=("Order_Id", "nunique"),
            seller_item_count=("Order_Item_Id", "count"),
            seller_revenue=("Price", "sum"),
            seller_avg_freight=("Freight_Value", "mean"),
            seller_late_rate=("late_delivery_flag", "mean"),
            seller_cancellation_rate=("cancelled_flag", "mean"),
            seller_avg_delivery_days=("delivery_days", "mean"),
            seller_median_delay_days=("delivery_delay_days", "median"),
        )
        .reset_index()
    )
    sf["seller_revenue_share_pct"] = sf["seller_revenue"] / sf["seller_revenue"].sum() * 100
    return sf.merge(sellers[["Seller_Id", "Seller_City", "Seller_State"]], on="Seller_Id", how="left")


def build_review_features(reviews: pd.DataFrame, order_features: pd.DataFrame) -> pd.DataFrame:
    rf = reviews.copy()
    rf["review_length"] = rf["Review_Text"].str.len().fillna(0).astype(int)
    rf["review_word_count"] = rf["Review_Text"].str.split().str.len().fillna(0).astype(int)
    rf["title_exists"] = rf["Review_Comment_Title"].str.strip().ne("").astype("int8")
    rf["sentiment_label"] = np.select(
        [rf["Review_Score"] >= 4, rf["Review_Score"] == 3, rf["Review_Score"] <= 2],
        ["positive", "neutral", "negative"],
        default="unknown",
    )
    rf["is_bad_review"] = rf["Review_Score"].le(2).astype("int8")
    rf["response_hours"] = (
        rf["Review_Answer_Timestamp"] - rf["Review_Creation_Date"]
    ).dt.total_seconds() / 3600

    linked = rf.merge(
        order_features[["Order_Id", "Order_Delivered_Customer_Date", "late_delivery_flag", "delivery_delay_days"]],
        on="Order_Id",
        how="left",
    )
    # A review created before delivery cannot be about the delivery experience.
    linked["review_before_delivery"] = (
        linked["Review_Creation_Date"] < linked["Order_Delivered_Customer_Date"]
    ).astype("int8")
    return linked.drop(columns=["Order_Delivered_Customer_Date"])


# --------------------------------------------------------------------------
# ML base table
# --------------------------------------------------------------------------
def assign_split(purchase_ts: pd.Series) -> pd.Series:
    """Chronological train / valid / test labels from cfg.TRAIN_END and cfg.VALID_END."""
    return pd.Series(
        np.select(
            [purchase_ts < cfg.TRAIN_END, purchase_ts < cfg.VALID_END],
            ["train", "valid"],
            default="test",
        ),
        index=purchase_ts.index,
        dtype="object",
    )


# Predictors available at the moment of purchase. This list *is* the feature
# contract; the leakage audit in modeling.py checks it against POST_OUTCOME_COLUMNS.
PURCHASE_TIME_FEATURES = [
    # basket
    "order_item_count",
    "order_product_count",
    "order_seller_count",
    "order_gmv",
    "avg_item_price",
    "max_item_price",
    "is_multi_seller",
    # physical
    "order_weight_g",
    "order_volume_cm3",
    "max_item_weight_g",
    # money / freight
    "order_freight_total",
    "freight_ratio",
    "freight_per_kg",
    "order_total_paid",
    "payment_count",
    "max_installments",
    # promise
    "estimated_delivery_days",
    "shipping_limit_days",
    # calendar
    "purchase_month_num",
    "purchase_weekday_num",
    "purchase_hour",
    "purchase_is_weekend",
    "days_since_dataset_start",
    # geography
    "distance_km",
    "distance_km_mean",
    "same_state_flag",
    "n_shipping_legs",
    # leakage-safe history
    "seller_prior_orders",
    "seller_prior_late_rate",
    "route_prior_orders",
    "route_prior_late_rate",
    "customer_prior_orders",
    "customer_prior_revenue",
    "is_repeat_purchase",
    # categorical
    "dominant_payment_type",
    "main_category",
    "Customer_State",
    "Seller_State",
    "distance_bucket",
]

TARGETS = ["late_delivery_flag", "delivery_days", "delivery_delay_days", "order_freight_total"]


def build_model_base(order_features: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Assemble the single ML-ready table: predictors + targets + split label."""
    seller_hist = as_of_reliability(order_features, items, key="Seller_Id", prefix="seller")
    route_hist = as_of_route_reliability(order_features)
    cust_hist = as_of_customer_history(order_features)

    base = (
        order_features.merge(seller_hist, on="Order_Id", how="left")
        .merge(route_hist, on="Order_Id", how="left")
        .merge(cust_hist, on="Order_Id", how="left")
    )
    base["purchase_month_num"] = base["Order_Purchase_Timestamp"].dt.month.astype("Int8")
    base["split"] = assign_split(base["Order_Purchase_Timestamp"])

    keep = (
        ["Order_Id", "Customer_Unique_Id", "Order_Purchase_Timestamp", "split", "Order_Status", "delivered_flag"]
        + [c for c in PURCHASE_TIME_FEATURES if c in base.columns]
        + [c for c in TARGETS if c in base.columns]
    )
    keep = list(dict.fromkeys(keep))
    return base[keep].sort_values("Order_Purchase_Timestamp").reset_index(drop=True)


def build_feature_tables(clean: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Run the whole feature phase. Returns (feature_tables, audits)."""
    orders = clean["olist_orders_dataset"]
    items = clean["olist_order_items_dataset"]
    payments = clean["olist_order_payments_dataset"]
    reviews = clean["olist_order_reviews_dataset"]
    customers = clean["olist_customers_dataset"]
    products = clean["olist_products_dataset"]
    sellers = clean["olist_sellers_dataset"]
    geolocation = clean["olist_geolocation_dataset"]

    order_geo = build_order_geo(orders, items, customers, sellers, geolocation)
    order_features = build_order_features(orders, items, payments, products, order_geo, customers)

    tables = {
        "order_features": order_features,
        "order_geo_features": order_geo,
        "customer_features": build_customer_features(order_features),
        "customer_order_history_as_of": as_of_customer_history(order_features),
        "product_features": build_product_features(items, products, order_features),
        "seller_features": build_seller_features(items, sellers, order_features),
        "review_features": build_review_features(reviews, order_features),
        "order_model_base": build_model_base(order_features, items),
    }
    audits = {"grain": grain_audit(tables), "reconciliation": feature_reconciliation(clean, tables)}
    return tables, audits


def grain_audit(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Declared grain vs actual uniqueness. A fanned-out join shows up here."""
    expected = {
        "order_features": "Order_Id",
        "order_geo_features": "Order_Id",
        "customer_features": "Customer_Unique_Id",
        "customer_order_history_as_of": "Order_Id",
        "product_features": "Product_Id",
        "seller_features": "Seller_Id",
        "review_features": "Order_Id",
        "order_model_base": "Order_Id",
    }
    rows = []
    for name, key in expected.items():
        df = tables[name]
        rows.append(
            {
                "table": name,
                "declared_grain": key,
                "rows": len(df),
                "unique_keys": df[key].nunique(),
                "status": "PASS" if len(df) == df[key].nunique() else "FAIL - grain broken",
            }
        )
    return pd.DataFrame(rows)


def feature_reconciliation(clean: Dict[str, pd.DataFrame], tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    items = clean["olist_order_items_dataset"]
    orders = clean["olist_orders_dataset"]
    of = tables["order_features"]
    rows = [
        ("unique orders", orders["Order_Id"].nunique(), of["Order_Id"].nunique()),
        ("item price total", items["Price"].sum(), of["order_gmv"].sum()),
        ("freight total", items["Freight_Value"].sum(), of["order_freight_total"].sum()),
        ("items", len(items), of["order_item_count"].sum()),
    ]
    recon = pd.DataFrame(rows, columns=["metric", "clean_tables", "feature_tables"])
    recon["diff"] = recon["feature_tables"] - recon["clean_tables"]
    recon["diff_pct"] = (recon["diff"] / recon["clean_tables"].replace(0, np.nan) * 100).round(5)
    recon["status"] = np.where(recon["diff_pct"].abs().fillna(0) < 0.01, "PASS", "REVIEW")
    return recon
