-- ============================================================================
-- Olist E-Commerce Delivery Analytics Schema
-- Dialect: SQLite / DuckDB / PostgreSQL compatible
-- ============================================================================

CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id VARCHAR(32) PRIMARY KEY,
    customer_unique_id VARCHAR(32) NOT NULL,
    customer_zip_code_prefix INTEGER NOT NULL,
    customer_city VARCHAR(64) NOT NULL,
    customer_state VARCHAR(4) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_sellers (
    seller_id VARCHAR(32) PRIMARY KEY,
    seller_zip_code_prefix INTEGER NOT NULL,
    seller_city VARCHAR(64) NOT NULL,
    seller_state VARCHAR(4) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_products (
    product_id VARCHAR(32) PRIMARY KEY,
    product_category_name VARCHAR(64),
    product_weight_g REAL,
    product_length_cm REAL,
    product_height_cm REAL,
    product_width_cm REAL
);

CREATE TABLE IF NOT EXISTS fact_orders (
    order_id VARCHAR(32) PRIMARY KEY,
    customer_id VARCHAR(32) NOT NULL,
    order_status VARCHAR(16) NOT NULL,
    order_purchase_timestamp TIMESTAMP NOT NULL,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP NOT NULL,
    late_delivery_flag INTEGER NOT NULL DEFAULT 0,
    delivery_lead_time_days REAL,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id)
);

CREATE TABLE IF NOT EXISTS fact_order_items (
    order_id VARCHAR(32) NOT NULL,
    order_item_id INTEGER NOT NULL,
    product_id VARCHAR(32) NOT NULL,
    seller_id VARCHAR(32) NOT NULL,
    shipping_limit_date TIMESTAMP NOT NULL,
    price REAL NOT NULL,
    freight_value REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES fact_orders(order_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (seller_id) REFERENCES dim_sellers(seller_id)
);

CREATE INDEX IF NOT EXISTS idx_order_purchase ON fact_orders(order_purchase_timestamp);
CREATE INDEX IF NOT EXISTS idx_order_cust ON fact_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_item_seller ON fact_order_items(seller_id);

