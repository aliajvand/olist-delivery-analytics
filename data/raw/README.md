# data/raw

Drop the nine Olist CSVs here, unzipped, with their original filenames:

```
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Licence: CC BY-NC-SA 4.0. Not redistributed in this repo.

Nothing in this project writes to this directory. Notebook 01 checks the row counts
against the canonical release, so a truncated or re-sampled download is caught
before it silently changes every number downstream.

No download yet? `python scripts/make_sample_data.py` builds a synthetic
Olist-shaped sample so the pipeline, notebooks and tests can be run immediately.
