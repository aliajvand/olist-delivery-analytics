# BI semantic model

Spec for the reporting layer. The .pbix is not in this repo; this is what it must
implement, and `bi_validation_aggregates.csv` is the Python truth every measure is
reconciled against.

## Star schema

Built on the parquet files in `data/processed/features/`.

**Fact table:** `order_features`, one row per `Order_Id`. Grain is asserted in
notebook 04, so no measure can fan out.

| Dimension | Source | Key | Relationship |
|---|---|---|---|
| Date | generated date table | `purchase_date` | 1:* to fact, marked as the date table |
| Customer | `customer_features` | `Customer_Unique_Id` | 1:* |
| Seller | `seller_features` | `Seller_Id` | *:* via order items - see note |
| Product category | `product_features` | `Product_Category_Name_English` | *:* via order items |
| Geography | `order_geo_features` | `Order_Id` | 1:1 |
| Payment type | `dominant_payment_type` | value | 1:* |

**Note on seller and product.** An order can contain several sellers and products,
so those two are many-to-many with the order fact. Do **not** join them directly to
`order_features`: use `olist_order_items_dataset` as a bridge, and never compute
order-level GMV through it or every multi-item order is double counted.

## Measures

| Measure | Definition | Python check |
|---|---|---|
| Orders | `DISTINCTCOUNT(order_features[Order_Id])` | `orders_total` |
| GMV proxy | `SUM(order_features[order_gmv])` | `gmv_proxy_brl` |
| Freight | `SUM(order_features[order_freight_total])` | `freight_total_brl` |
| AOV | `[GMV proxy] / [Orders]` | `aov_brl` |
| Items | `SUM(order_features[order_item_count])` | `items_total` |
| Delivered orders | `CALCULATE([Orders], order_features[delivered_flag] = 1)` | `orders_delivered` |
| Late delivery rate | `AVERAGE(order_features[late_delivery_flag])` - **ignores blanks by design** | `late_delivery_rate` |
| Cancellation rate | `AVERAGE(order_features[cancelled_flag])` | `cancellation_rate` |
| Avg delivery days | `AVERAGE(order_features[delivery_days])` | `avg_delivery_days` |
| Median delay days | `MEDIAN(order_features[delivery_delay_days])` | `median_delay_days` |
| Avg review score | `AVERAGE(review_features[Review_Score])` | `avg_review_score` |
| Bad review rate | `AVERAGE(review_features[is_bad_review])` | `bad_review_rate` |
| Unique customers | `DISTINCTCOUNT(order_features[Customer_Unique_Id])` | `unique_customers` |
| Repeat customer rate | `AVERAGE(customer_features[repeat_customer_flag])` | `repeat_customer_rate` |

## Three rules the report must follow

1. **Never label GMV as "revenue" or "profit".** The dataset has no cost of goods,
   commission or marketing spend. See EDA Q15.
2. **Late delivery rate is computed over delivered orders only.** `late_delivery_flag`
   is blank for undelivered orders, and DAX averages ignore blanks - which is the
   correct behaviour. Do not `COALESCE` it to 0; that invents on-time deliveries.
3. **Exclude the incomplete final month from every trend.** The extract boundary
   truncates in-flight deliveries, so the last month's delivered share collapses for
   reasons that have nothing to do with operations.

## Suggested pages

1. **Executive** - orders, GMV proxy, AOV, late rate, avg review score, monthly trend.
2. **Delivery performance** - funnel, delay distribution, late rate by state and by
   distance band, promised vs actual days.
3. **Seller scorecard** - reliability vs volume, bubble-sized by GMV, ranked by
   *excess late orders* rather than by rate so the list is actionable.
4. **Customer** - RFM segments, repeat rate, value concentration.
5. **Assumptions** - the limitations from `README.md`, on the report itself. Nobody
   reads the appendix.

## Reconciliation procedure

1. Run `notebooks/03_eda.ipynb` to regenerate `bi_validation_aggregates.csv`.
2. Build the same measures with no filters applied.
3. Paste them into the `bi_value` column and set `match`.
4. Anything off by more than 0.01% is a bug in the DAX or the model - almost always
   a many-to-many join fanning out the fact table.
