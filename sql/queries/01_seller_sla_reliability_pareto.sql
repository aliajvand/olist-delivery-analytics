-- ============================================================================
-- Business Question 1: Seller Fulfillment SLA & Late Rate Pareto Analysis
-- Technique: CTEs, Window Functions (Cumulative Sums, Dense Rank), Aggregations
-- ============================================================================

WITH seller_orders AS (
    SELECT
        s.seller_id,
        s.seller_state,
        COUNT(DISTINCT o.order_id) AS total_shipped_orders,
        SUM(o.late_delivery_flag) AS total_late_orders,
        SUM(i.price) AS total_gmv,
        AVG(o.delivery_lead_time_days) AS avg_delivery_days
    FROM fact_order_items i
    JOIN fact_orders o ON i.order_id = o.order_id
    JOIN dim_sellers s ON i.seller_id = s.seller_id
    WHERE o.order_status = 'delivered'
    GROUP BY s.seller_id, s.seller_state
),
seller_metrics AS (
    SELECT
        seller_id,
        seller_state,
        total_shipped_orders,
        total_late_orders,
        ROUND(total_gmv, 2) AS gmv,
        ROUND(avg_delivery_days, 1) AS avg_days,
        ROUND(total_late_orders * 100.0 / NULLIF(total_shipped_orders, 0), 2) AS late_rate_pct,
        SUM(total_late_orders) OVER (ORDER BY total_late_orders DESC) * 100.0 / NULLIF(SUM(total_late_orders) OVER (), 0) AS cumulative_late_share,
        DENSE_RANK() OVER (ORDER BY total_late_orders DESC) AS late_volume_rank
    FROM seller_orders
    WHERE total_shipped_orders >= 20
)
SELECT
    late_volume_rank,
    seller_id,
    seller_state,
    total_shipped_orders,
    total_late_orders,
    late_rate_pct,
    gmv,
    avg_days,
    ROUND(cumulative_late_share, 2) AS cumulative_late_pct
FROM seller_metrics
ORDER BY late_volume_rank
LIMIT 30;
