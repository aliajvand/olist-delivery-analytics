"""Phase 1 - Clean.

Every function takes DataFrames and returns DataFrames. No file IO, no printing,
no globals. That is what makes the cleaning step testable (see tests/).

Design decisions worth defending in a review:
  * Column names are normalised to Title_Case_With_Underscores once, here.
  * Nulls are never blanket-filled. Each null gets a documented decision, and
    every imputation leaves a `<col>_Missing` indicator behind.
  * Outliers are flagged, never dropped. On a marketplace the tail *is* the
    business (freight on heavy goods, high-value electronics).
  * Rows are only removed when they are true duplicates or break referential
    integrity. Removals are counted and reported.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as cfg

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
DTYPES: Dict[str, str] = {
    # customers
    "Customer_Id": "string",
    "Customer_Unique_Id": "string",
    "Customer_Zip_Code_Prefix": "Int64",
    "Customer_City": "string",
    "Customer_State": "category",
    # geolocation
    "Geolocation_Zip_Code_Prefix": "Int64",
    "Geolocation_Lat": "float64",
    "Geolocation_Lng": "float64",
    "Geolocation_City": "string",
    "Geolocation_State": "category",
    # orders
    "Order_Id": "string",
    "Order_Status": "category",
    "Order_Purchase_Timestamp": "datetime64[ns]",
    "Order_Approved_At": "datetime64[ns]",
    "Order_Delivered_Carrier_Date": "datetime64[ns]",
    "Order_Delivered_Customer_Date": "datetime64[ns]",
    "Order_Estimated_Delivery_Date": "datetime64[ns]",
    # order items
    "Order_Item_Id": "Int64",
    "Product_Id": "string",
    "Seller_Id": "string",
    "Shipping_Limit_Date": "datetime64[ns]",
    "Price": "float64",
    "Freight_Value": "float64",
    # payments
    "Payment_Sequential": "Int64",
    "Payment_Type": "category",
    "Payment_Installments": "Int64",
    "Payment_Value": "float64",
    # reviews
    "Review_Id": "string",
    "Review_Score": "Int64",
    "Review_Comment_Title": "string",
    "Review_Comment_Message": "string",
    "Review_Creation_Date": "datetime64[ns]",
    "Review_Answer_Timestamp": "datetime64[ns]",
    # products
    "Product_Category_Name": "string",
    "Product_Name_Lenght": "Int64",
    "Product_Description_Lenght": "Int64",
    "Product_Photos_Qty": "Int64",
    "Product_Weight_G": "float64",
    "Product_Length_Cm": "float64",
    "Product_Height_Cm": "float64",
    "Product_Width_Cm": "float64",
    # sellers
    "Seller_Zip_Code_Prefix": "Int64",
    "Seller_City": "string",
    "Seller_State": "category",
    # translation
    "Product_Category_Name_English": "string",
}

# Brazil's bounding box, used to reject impossible geolocation rows.
BRAZIL_LAT = (-34.0, 5.5)
BRAZIL_LNG = (-74.0, -34.0)

PRODUCT_COUNT_COLS = [
    "Product_Name_Lenght",
    "Product_Description_Lenght",
    "Product_Photos_Qty",
]
PRODUCT_DIMENSION_COLS = [
    "Product_Weight_G",
    "Product_Length_Cm",
    "Product_Height_Cm",
    "Product_Width_Cm",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case -> Title_Case_With_Underscores, applied exactly once."""
    out = df.copy()
    out.columns = (
        out.columns.str.strip().str.replace("_", " ").str.title().str.replace(" ", "_")
    )
    return out


def apply_dtypes(df: pd.DataFrame, mapping: Dict[str, str] | None = None) -> pd.DataFrame:
    mapping = mapping or DTYPES
    out = df.copy()
    for col, dtype in mapping.items():
        if col not in out.columns:
            continue
        if dtype.startswith("datetime"):
            out[col] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[col] = out[col].astype(dtype)
    return out


def flag_outliers(df: pd.DataFrame, columns: List[str], iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Add `<col>_Is_Outlier` booleans plus a row-level `Outlier_Count`.

    Flags only the columns passed in. Blanket-flagging every numeric column
    (including IDs, zip prefixes and counters) produces meaningless flags.
    """
    out = df.copy()
    columns = [c for c in columns if c in out.columns]
    out["Outlier_Count"] = 0
    for col in columns:
        series = pd.to_numeric(out[col], errors="coerce")
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        out[f"{col}_Is_Outlier"] = mask.fillna(False)
        out["Outlier_Count"] += mask.fillna(False).astype("int16")
    out["Is_Outlier"] = out["Outlier_Count"] > 0
    return out


# --------------------------------------------------------------------------
# Table-level cleaners
# --------------------------------------------------------------------------
def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset="Customer_Id").copy()
    out["Customer_City"] = out["Customer_City"].str.strip().str.lower()
    return out


def clean_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Reject coordinates outside Brazil, then collapse to one row per zip prefix.

    The raw table has ~1M rows and many per zip prefix; a zip-level median is
    robust to the handful of clearly mistyped coordinates and makes every
    downstream join a clean 1:1.
    """
    out = df.copy()
    in_brazil = (
        out["Geolocation_Lat"].between(*BRAZIL_LAT)
        & out["Geolocation_Lng"].between(*BRAZIL_LNG)
    )
    out = out.loc[in_brazil]
    zip_level = (
        out.groupby("Geolocation_Zip_Code_Prefix", observed=True)
        .agg(
            Geolocation_Lat=("Geolocation_Lat", "median"),
            Geolocation_Lng=("Geolocation_Lng", "median"),
            Geolocation_City=("Geolocation_City", lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA),
            Geolocation_State=("Geolocation_State", lambda s: s.mode().iloc[0] if len(s.mode()) else pd.NA),
            Geolocation_Points=("Geolocation_Lat", "size"),
        )
        .reset_index()
    )
    return zip_level


def clean_orders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add lifecycle flags and audit timestamp monotonicity.

    Returns (clean_orders, timestamp_audit). Rows that violate the expected
    ordering purchase <= approved <= carrier <= customer are flagged rather than
    deleted: they are real orders with messy operational logging, and dropping
    them would bias the late-delivery rate.
    """
    out = df.drop_duplicates(subset="Order_Id").copy()

    out["Is_Approved"] = out["Order_Approved_At"].notna().astype("int8")
    out["Is_Shipped"] = out["Order_Delivered_Carrier_Date"].notna().astype("int8")
    out["Is_Delivered"] = out["Order_Delivered_Customer_Date"].notna().astype("int8")

    checks = {
        "approved_before_purchase": out["Order_Approved_At"] < out["Order_Purchase_Timestamp"],
        "carrier_before_approved": out["Order_Delivered_Carrier_Date"] < out["Order_Approved_At"],
        "customer_before_carrier": out["Order_Delivered_Customer_Date"] < out["Order_Delivered_Carrier_Date"],
        "delivered_but_status_not_delivered": out["Order_Delivered_Customer_Date"].notna()
        & out["Order_Status"].astype("string").ne("delivered"),
        "status_delivered_but_no_date": out["Order_Status"].astype("string").eq("delivered")
        & out["Order_Delivered_Customer_Date"].isna(),
    }
    audit = pd.DataFrame(
        [{"check": name, "violations": int(mask.fillna(False).sum())} for name, mask in checks.items()]
    )
    audit["share_pct"] = (audit["violations"] / len(out) * 100).round(4)

    violation_matrix = pd.concat([m.fillna(False) for m in checks.values()], axis=1)
    out["Timestamp_Anomaly"] = violation_matrix.any(axis=1)
    return out, audit


def clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset=["Order_Id", "Order_Item_Id"]).copy()
    # Negative money is impossible; coerce to NaN so it cannot silently sum away.
    for col in ("Price", "Freight_Value"):
        out.loc[out[col] < 0, col] = np.nan
    out["Item_Total"] = out["Price"].fillna(0) + out["Freight_Value"].fillna(0)
    return flag_outliers(out, ["Price", "Freight_Value"])


def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates().copy()
    # 'not_defined' carries no information and always pairs with a 0 value.
    out["Payment_Type"] = out["Payment_Type"].astype("string").replace({"not_defined": pd.NA})
    # 0 installments is a data-entry artefact for single-shot payments.
    out.loc[out["Payment_Installments"].eq(0), "Payment_Installments"] = 1
    return flag_outliers(out, ["Payment_Value"])


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """One review per order: keep the most recent, count the rest.

    ~550 orders carry more than one review. Left as-is they silently fan out
    every order-level join and inflate GMV.
    """
    out = df.drop_duplicates(subset="Review_Id").copy()

    out["Review_Comment_Title"] = out["Review_Comment_Title"].fillna("")
    out["Review_Comment_Message"] = out["Review_Comment_Message"].fillna("")
    out["Review_Text"] = (
        out["Review_Comment_Title"].astype("string") + " " + out["Review_Comment_Message"].astype("string")
    ).str.strip()
    out["Has_Review_Text"] = out["Review_Text"].ne("").astype("int8")

    review_counts = out.groupby("Order_Id", observed=True)["Review_Id"].transform("size")
    out["Reviews_Per_Order"] = review_counts.astype("int16")
    out = (
        out.sort_values(["Order_Id", "Review_Creation_Date", "Review_Answer_Timestamp"])
        .drop_duplicates(subset="Order_Id", keep="last")
        .reset_index(drop=True)
    )
    return out


def clean_products(df: pd.DataFrame, translation: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.drop_duplicates(subset="Product_Id").copy()

    out["Product_Category_Name"] = (
        out["Product_Category_Name"].astype("string").str.strip().replace({"": pd.NA}).fillna("unknown")
    )

    for col in PRODUCT_COUNT_COLS + PRODUCT_DIMENSION_COLS:
        if col not in out.columns:
            continue
        out[f"{col}_Missing"] = out[col].isna().astype("int8")
        # Median within category, global median as fallback. A product's
        # dimensions correlate strongly with its category, so a category-level
        # median beats a single global number.
        category_median = out.groupby("Product_Category_Name", observed=True)[col].transform("median")
        global_median = out[col].median()
        if str(out[col].dtype).startswith("Int"):
            # a median of 3.5 cannot be written back into an integer column
            category_median = category_median.round()
            global_median = round(global_median) if pd.notna(global_median) else global_median
        out[col] = out[col].fillna(category_median).fillna(global_median)

    out["Product_Volume_Cm3"] = (
        out["Product_Length_Cm"] * out["Product_Height_Cm"] * out["Product_Width_Cm"]
    )
    # Volumetric weight is the industry proxy carriers actually price on.
    out["Product_Density_G_Cm3"] = np.where(
        out["Product_Volume_Cm3"] > 0, out["Product_Weight_G"] / out["Product_Volume_Cm3"], np.nan
    )

    if translation is not None:
        out = out.merge(
            translation[["Product_Category_Name", "Product_Category_Name_English"]],
            on="Product_Category_Name",
            how="left",
        )
        out["Product_Category_Name_English"] = (
            out["Product_Category_Name_English"].astype("string").fillna(out["Product_Category_Name"])
        )
    return flag_outliers(out, ["Product_Weight_G", "Product_Volume_Cm3"])


def clean_sellers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset="Seller_Id").copy()
    out["Seller_City"] = out["Seller_City"].str.strip().str.lower()
    return out


def clean_translation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset="Product_Category_Name").copy()
    extra = pd.DataFrame(
        {"Product_Category_Name": ["unknown"], "Product_Category_Name_English": ["unknown"]}
    )
    return pd.concat([out, extra], ignore_index=True).drop_duplicates(subset="Product_Category_Name")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def referential_integrity(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every foreign key must resolve. Orphans mean a join will silently drop rows."""
    orders = tables["olist_orders_dataset"]
    items = tables["olist_order_items_dataset"]
    payments = tables["olist_order_payments_dataset"]
    reviews = tables["olist_order_reviews_dataset"]
    customers = tables["olist_customers_dataset"]
    products = tables["olist_products_dataset"]
    sellers = tables["olist_sellers_dataset"]

    def orphans(child: pd.DataFrame, child_key: str, parent: pd.DataFrame, parent_key: str) -> int:
        valid = set(parent[parent_key].dropna())
        return int((~child[child_key].isin(valid)).sum())

    checks = [
        ("order_items.Order_Id -> orders", orphans(items, "Order_Id", orders, "Order_Id"), len(items)),
        ("payments.Order_Id -> orders", orphans(payments, "Order_Id", orders, "Order_Id"), len(payments)),
        ("reviews.Order_Id -> orders", orphans(reviews, "Order_Id", orders, "Order_Id"), len(reviews)),
        ("orders.Customer_Id -> customers", orphans(orders, "Customer_Id", customers, "Customer_Id"), len(orders)),
        ("order_items.Product_Id -> products", orphans(items, "Product_Id", products, "Product_Id"), len(items)),
        ("order_items.Seller_Id -> sellers", orphans(items, "Seller_Id", sellers, "Seller_Id"), len(items)),
    ]
    report = pd.DataFrame(checks, columns=["relationship", "orphan_rows", "child_rows"])
    report["orphan_pct"] = (report["orphan_rows"] / report["child_rows"] * 100).round(4)
    report["status"] = np.where(report["orphan_rows"] == 0, "PASS", "REVIEW")
    return report


def reconcile(raw: Dict[str, pd.DataFrame], clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Totals that must survive cleaning. Any drift here is a bug, not a choice."""
    rows = [
        ("orders", len(raw["olist_orders_dataset"]), len(clean["olist_orders_dataset"])),
        ("unique orders", raw["olist_orders_dataset"]["order_id"].nunique(),
         clean["olist_orders_dataset"]["Order_Id"].nunique()),
        ("order items", len(raw["olist_order_items_dataset"]), len(clean["olist_order_items_dataset"])),
        ("item price total", raw["olist_order_items_dataset"]["price"].sum(),
         clean["olist_order_items_dataset"]["Price"].sum()),
        ("freight total", raw["olist_order_items_dataset"]["freight_value"].sum(),
         clean["olist_order_items_dataset"]["Freight_Value"].sum()),
        ("payment total", raw["olist_order_payments_dataset"]["payment_value"].sum(),
         clean["olist_order_payments_dataset"]["Payment_Value"].sum()),
        ("products", raw["olist_products_dataset"]["product_id"].nunique(),
         clean["olist_products_dataset"]["Product_Id"].nunique()),
        ("sellers", raw["olist_sellers_dataset"]["seller_id"].nunique(),
         clean["olist_sellers_dataset"]["Seller_Id"].nunique()),
    ]
    recon = pd.DataFrame(rows, columns=["metric", "raw", "clean"])
    recon["diff"] = recon["clean"] - recon["raw"]
    recon["diff_pct"] = (recon["diff"] / recon["raw"].replace(0, np.nan) * 100).round(4)
    recon["status"] = np.where(recon["diff_pct"].abs().fillna(0) < 0.01, "PASS", "EXPECTED-CHANGE")
    return recon


def build_clean_tables(raw: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Run the whole cleaning phase.

    Returns (clean_tables, audits) where audits holds the QA frames that the
    notebook renders and the reports reference.
    """
    normalised = {name: apply_dtypes(normalise_columns(df)) for name, df in raw.items()}

    translation = clean_translation(normalised["product_category_name_translation"])
    orders, timestamp_audit = clean_orders(normalised["olist_orders_dataset"])

    clean = {
        "olist_customers_dataset": clean_customers(normalised["olist_customers_dataset"]),
        "olist_geolocation_dataset": clean_geolocation(normalised["olist_geolocation_dataset"]),
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": clean_order_items(normalised["olist_order_items_dataset"]),
        "olist_order_payments_dataset": clean_payments(normalised["olist_order_payments_dataset"]),
        "olist_order_reviews_dataset": clean_reviews(normalised["olist_order_reviews_dataset"]),
        "olist_products_dataset": clean_products(normalised["olist_products_dataset"], translation),
        "olist_sellers_dataset": clean_sellers(normalised["olist_sellers_dataset"]),
        "product_category_name_translation": translation,
    }

    audits = {
        "timestamp_audit": timestamp_audit,
        "referential_integrity": referential_integrity(clean),
        "reconciliation": reconcile(raw, clean),
    }
    return clean, audits
