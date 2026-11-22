"""Cleaning must be lossless where it claims to be, and explicit where it is not."""
import pandas as pd

from olist import cleaning


def test_column_names_are_normalised_once(clean):
    for name, df in clean.items():
        for col in df.columns:
            assert col == col.strip(), f"{name}.{col} has whitespace"
            assert " " not in col, f"{name}.{col} contains a space"
            assert col[0].isupper(), f"{name}.{col} is not Title_Case"


def test_normalise_columns_is_idempotent(raw):
    once = cleaning.normalise_columns(raw["olist_orders_dataset"])
    twice = cleaning.normalise_columns(once)
    assert list(once.columns) == list(twice.columns)


def test_money_totals_survive_cleaning(raw, clean):
    raw_price = raw["olist_order_items_dataset"]["price"].sum()
    clean_price = clean["olist_order_items_dataset"]["Price"].sum()
    assert abs(clean_price - raw_price) < 0.01


def test_primary_keys_are_unique(clean):
    keys = {
        "olist_orders_dataset": "Order_Id",
        "olist_customers_dataset": "Customer_Id",
        "olist_products_dataset": "Product_Id",
        "olist_sellers_dataset": "Seller_Id",
        "olist_geolocation_dataset": "Geolocation_Zip_Code_Prefix",
        "olist_order_reviews_dataset": "Order_Id",
    }
    for table, key in keys.items():
        assert clean[table][key].is_unique, f"{table}.{key} is not unique"


def test_reviews_are_one_per_order(raw, clean):
    """The raw data has orders with two reviews. Left alone they fan out joins."""
    reviews = clean["olist_order_reviews_dataset"]
    assert reviews["Order_Id"].is_unique
    assert (reviews["Reviews_Per_Order"] > 1).any(), "fixture should contain duplicate reviews"


def test_geolocation_collapsed_to_one_row_per_zip(clean):
    geo = clean["olist_geolocation_dataset"]
    assert geo["Geolocation_Zip_Code_Prefix"].is_unique
    assert geo["Geolocation_Lat"].between(-34, 5.5).all()
    assert geo["Geolocation_Lng"].between(-74, -34).all()


def test_no_referential_integrity_violations(clean_audits):
    report = clean_audits["referential_integrity"]
    assert (report["orphan_rows"] == 0).all(), report.to_string(index=False)


def test_imputation_leaves_an_indicator(clean):
    products = clean["olist_products_dataset"]
    assert "Product_Weight_G_Missing" in products.columns
    assert products["Product_Weight_G"].notna().all()
    assert products["Product_Weight_G_Missing"].sum() > 0


def test_outliers_are_flagged_not_dropped(raw, clean):
    items = clean["olist_order_items_dataset"]
    assert len(items) == raw["olist_order_items_dataset"].drop_duplicates(
        subset=["order_id", "order_item_id"]
    ).shape[0]
    assert "Price_Is_Outlier" in items.columns


def test_flag_outliers_only_touches_requested_columns():
    df = pd.DataFrame({"Price": [1, 2, 3, 1000], "Zip": [1000, 2000, 3000, 90000]})
    out = cleaning.flag_outliers(df, ["Price"])
    assert "Price_Is_Outlier" in out.columns
    assert "Zip_Is_Outlier" not in out.columns
    assert out["Price_Is_Outlier"].sum() == 1
