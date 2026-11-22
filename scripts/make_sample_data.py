"""Generate a small Olist-SHAPED synthetic dataset for smoke tests and CI.

This is NOT the Olist data and must never be used for analysis or reported
numbers. It exists so `run_pipeline.py`, the notebooks and the tests can be
executed on a machine that does not have the 120 MB Kaggle download yet.

Usage:
    python scripts/make_sample_data.py --out data/sample_raw --orders 8000
    OLIST_DATA_DIR=data_sample python scripts/run_pipeline.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "PE", "CE", "DF", "ES", "PA", "AM"]
STATE_LATLNG = {
    "SP": (-23.55, -46.63), "RJ": (-22.91, -43.17), "MG": (-19.92, -43.94),
    "RS": (-30.03, -51.23), "PR": (-25.43, -49.27), "SC": (-27.59, -48.55),
    "BA": (-12.97, -38.50), "GO": (-16.68, -49.25), "PE": (-8.05, -34.88),
    "CE": (-3.73, -38.52), "DF": (-15.79, -47.88), "ES": (-20.32, -40.34),
    "PA": (-1.46, -48.50), "AM": (-3.12, -60.02),
}
STATE_WEIGHTS = np.array([42, 13, 12, 5.5, 5, 3.5, 3.4, 2, 1.7, 1.5, 2.2, 2, 1, 0.9])
STATE_WEIGHTS = STATE_WEIGHTS / STATE_WEIGHTS.sum()
CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
    "computers_accessories", "housewares", "watches_gifts", "telephony",
    "auto", "toys", "garden_tools", "cool_stuff",
]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]
PAYMENT_WEIGHTS = [0.74, 0.19, 0.05, 0.02]


def hex_ids(rng: np.random.Generator, n: int, prefix: str) -> np.ndarray:
    return np.array([f"{prefix}{i:027x}" for i in rng.choice(16**7, size=n, replace=False)])


def build(n_orders: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_customers = int(n_orders * 0.97)
    n_sellers = max(30, n_orders // 30)
    n_products = max(100, n_orders // 4)

    # ---- geolocation ----------------------------------------------------
    zips = rng.choice(np.arange(1000, 99000), size=n_orders // 2 + 500, replace=False)
    zip_state = rng.choice(STATES, size=len(zips), p=STATE_WEIGHTS)
    geo_rows = []
    for zip_code, state in zip(zips, zip_state):
        base_lat, base_lng = STATE_LATLNG[state]
        for _ in range(rng.integers(1, 4)):
            geo_rows.append(
                {
                    "geolocation_zip_code_prefix": int(zip_code),
                    "geolocation_lat": base_lat + rng.normal(0, 1.2),
                    "geolocation_lng": base_lng + rng.normal(0, 1.2),
                    "geolocation_city": f"city_{zip_code % 400}",
                    "geolocation_state": state,
                }
            )
    geolocation = pd.DataFrame(geo_rows)

    zip_lookup = pd.DataFrame({"zip": zips, "state": zip_state})

    # ---- customers / sellers / products ---------------------------------
    cust_zip = zip_lookup.sample(n_customers, replace=True, random_state=seed).reset_index(drop=True)
    customers = pd.DataFrame(
        {
            "customer_id": hex_ids(rng, n_customers, "c"),
            "customer_unique_id": hex_ids(rng, n_customers, "u"),
            "customer_zip_code_prefix": cust_zip["zip"].to_numpy(),
            "customer_city": [f"city_{z % 400}" for z in cust_zip["zip"]],
            "customer_state": cust_zip["state"].to_numpy(),
        }
    )
    # ~3% of customers order twice
    repeat_idx = rng.choice(n_customers, size=int(n_customers * 0.03), replace=False)
    customers.loc[repeat_idx, "customer_unique_id"] = customers.loc[
        rng.choice(repeat_idx, size=len(repeat_idx)), "customer_unique_id"
    ].to_numpy()

    seller_zip = zip_lookup.sample(n_sellers, replace=True, random_state=seed + 1).reset_index(drop=True)
    sellers = pd.DataFrame(
        {
            "seller_id": hex_ids(rng, n_sellers, "s"),
            "seller_zip_code_prefix": seller_zip["zip"].to_numpy(),
            "seller_city": [f"city_{z % 400}" for z in seller_zip["zip"]],
            "seller_state": seller_zip["state"].to_numpy(),
        }
    )
    # persistent seller quality, so the as-of reliability feature has signal
    seller_quality = pd.Series(rng.beta(2, 12, n_sellers), index=sellers["seller_id"])

    weight = rng.lognormal(6.7, 1.1, n_products)
    products = pd.DataFrame(
        {
            "product_id": hex_ids(rng, n_products, "p"),
            "product_category_name": rng.choice(CATEGORIES, n_products),
            "product_name_lenght": rng.integers(20, 70, n_products).astype(float),
            "product_description_lenght": rng.integers(100, 3000, n_products).astype(float),
            "product_photos_qty": rng.integers(1, 8, n_products).astype(float),
            "product_weight_g": weight,
            "product_length_cm": rng.integers(15, 90, n_products).astype(float),
            "product_height_cm": rng.integers(5, 60, n_products).astype(float),
            "product_width_cm": rng.integers(10, 70, n_products).astype(float),
        }
    )
    for col in ["product_category_name", "product_weight_g", "product_photos_qty"]:
        products.loc[rng.choice(n_products, size=max(1, n_products // 60), replace=False), col] = np.nan

    translation = pd.DataFrame(
        {"product_category_name": CATEGORIES, "product_category_name_english": CATEGORIES}
    )

    # ---- orders ---------------------------------------------------------
    order_ids = hex_ids(rng, n_orders, "o")
    start = pd.Timestamp("2016-10-01")
    # rising volume over 24 months, like the real thing
    day_offset = np.sort((rng.beta(2.4, 1.5, n_orders) * 720).astype(int))
    purchase = start + pd.to_timedelta(day_offset, unit="D") + pd.to_timedelta(
        rng.integers(0, 24 * 3600, n_orders), unit="s"
    )

    cust_pick = rng.choice(n_customers, n_orders, replace=True)
    seller_pick = rng.choice(n_sellers, n_orders, replace=True)

    cust_state = customers["customer_state"].to_numpy()[cust_pick]
    sell_state = sellers["seller_state"].to_numpy()[seller_pick]
    cust_ll = np.array([STATE_LATLNG[s] for s in cust_state])
    sell_ll = np.array([STATE_LATLNG[s] for s in sell_state])
    distance = np.sqrt(((cust_ll - sell_ll) ** 2).sum(axis=1)) * 111 + rng.normal(0, 40, n_orders)
    distance = np.clip(distance, 5, None)

    quality = seller_quality.to_numpy()[seller_pick]

    approved = purchase + pd.to_timedelta(np.abs(rng.normal(10, 12, n_orders)), unit="h")
    handling = np.abs(rng.normal(2.6, 2.2, n_orders)) + quality * 9
    transit = 1.6 + distance / 340 + rng.gamma(1.7, 1.5, n_orders) + quality * 6
    carrier = approved + pd.to_timedelta(handling, unit="D")
    delivered = carrier + pd.to_timedelta(transit, unit="D")
    # the promise is generous and scales weakly with distance
    promise_days = 12 + distance / 260 + rng.normal(0, 2.5, n_orders)
    estimated = (purchase + pd.to_timedelta(np.clip(promise_days, 4, None), unit="D")).normalize()

    status = np.full(n_orders, "delivered", dtype=object)
    status[rng.random(n_orders) < 0.006] = "canceled"
    status[rng.random(n_orders) < 0.012] = "shipped"
    # nothing is delivered after the data cutoff
    cutoff = pd.Timestamp("2018-10-17")
    status[delivered > cutoff] = "shipped"

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customers["customer_id"].to_numpy()[cust_pick],
            "order_status": status,
            "order_purchase_timestamp": purchase,
            "order_approved_at": approved,
            "order_delivered_carrier_date": carrier,
            "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated,
        }
    )
    not_delivered = orders["order_status"] != "delivered"
    orders.loc[not_delivered, "order_delivered_customer_date"] = pd.NaT
    orders.loc[orders["order_status"] == "canceled", ["order_approved_at", "order_delivered_carrier_date"]] = pd.NaT

    # ---- order items ----------------------------------------------------
    n_items = rng.choice([1, 1, 1, 2, 2, 3, 5], n_orders)
    item_rows = []
    for idx, (oid, count) in enumerate(zip(order_ids, n_items)):
        for item_no in range(1, count + 1):
            pid_idx = rng.integers(0, n_products)
            price = round(float(rng.lognormal(4.3, 0.85)), 2)
            freight = round(float(6 + distance[idx] / 130 + rng.gamma(1.6, 3.0)), 2)
            item_rows.append(
                {
                    "order_id": oid,
                    "order_item_id": item_no,
                    "product_id": products["product_id"].iloc[pid_idx],
                    "seller_id": sellers["seller_id"].iloc[seller_pick[idx]],
                    "shipping_limit_date": (purchase[idx] + pd.Timedelta(days=int(rng.integers(2, 9)))),
                    "price": price,
                    "freight_value": freight,
                }
            )
    order_items = pd.DataFrame(item_rows)

    order_value = order_items.groupby("order_id")[["price", "freight_value"]].sum().sum(axis=1)

    # ---- payments -------------------------------------------------------
    pay_type = rng.choice(PAYMENT_TYPES, n_orders, p=PAYMENT_WEIGHTS)
    installments = np.where(pay_type == "credit_card", rng.integers(1, 13, n_orders), 1)
    payments = pd.DataFrame(
        {
            "order_id": order_ids,
            "payment_sequential": 1,
            "payment_type": pay_type,
            "payment_installments": installments,
            "payment_value": order_value.reindex(order_ids).fillna(0).round(2).to_numpy(),
        }
    )
    # some orders split payment across two records
    split = payments.sample(frac=0.05, random_state=seed).copy()
    split["payment_sequential"] = 2
    split["payment_type"] = "voucher"
    split["payment_value"] = (split["payment_value"] * 0.3).round(2)
    payments = pd.concat([payments, split], ignore_index=True)

    # ---- reviews --------------------------------------------------------
    delay = (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.days
    base_score = np.where(delay.fillna(0) > 0, rng.normal(2.2, 1.1, n_orders), rng.normal(4.4, 0.85, n_orders))
    score = np.clip(np.round(base_score), 1, 5).astype(int)
    has_review = rng.random(n_orders) < 0.97
    reviews = pd.DataFrame(
        {
            "review_id": hex_ids(rng, n_orders, "r"),
            "order_id": order_ids,
            "review_score": score,
            "review_comment_title": np.where(rng.random(n_orders) < 0.12, "titulo", None).astype(object),
            "review_comment_message": np.where(
                rng.random(n_orders) < 0.42,
                np.where(score <= 2, "produto atrasou muito e nao recebi resposta do vendedor", "otimo produto"),
                None,
            ).astype(object),
            "review_creation_date": orders["order_delivered_customer_date"].fillna(
                orders["order_purchase_timestamp"] + pd.Timedelta(days=20)
            ),
            "review_answer_timestamp": orders["order_delivered_customer_date"].fillna(
                orders["order_purchase_timestamp"] + pd.Timedelta(days=20)
            )
            + pd.to_timedelta(rng.integers(1, 96, n_orders), unit="h"),
        }
    )[has_review].reset_index(drop=True)
    # a few orders get two reviews, exactly like the real data
    dupes = reviews.sample(frac=0.006, random_state=seed).copy()
    dupes["review_id"] = hex_ids(rng, len(dupes), "r")
    reviews = pd.concat([reviews, dupes], ignore_index=True)

    return {
        "olist_customers_dataset": customers,
        "olist_geolocation_dataset": geolocation,
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": order_items,
        "olist_order_payments_dataset": payments,
        "olist_order_reviews_dataset": reviews,
        "olist_products_dataset": products,
        "olist_sellers_dataset": sellers,
        "product_category_name_translation": translation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample_raw", help="output directory for the CSVs")
    parser.add_argument("--orders", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in build(args.orders, args.seed).items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{path}  {len(df):>8,} rows")
    print(f"\nSynthetic sample only - never report numbers from this. Wrote to {out.resolve()}")


if __name__ == "__main__":
    main()
