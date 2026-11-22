-- ============================================================================
-- Business Question 2: Interstate Route Logistics Bottlenecks
-- Technique: Multi-table Joins, Origin-Destination Matrix, Route Aggregation
-- ============================================================================

WITH route_stats AS (
    SELECT
        s.seller_state AS origin_state,
        c.customer_state AS destination_state,
        COUNT(DISTINCT o.order_id) AS route_order_volume,
        SUM(o.late_delivery_flag) AS route_late_count,
        AVG(o.delivery_lead_time_days) AS avg_transit_days,
        AVG(i.freight_value) AS avg_freight_cost
    FROM fact_orders o
    JOIN fact_order_items i ON o.order_id = i.order_id
    JOIN dim_sellers s ON i.seller_id = s.seller_id
    JOIN dim_customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY s.seller_state, c.customer_state
)
SELECT
    origin_state,
    destination_state,
    route_order_volume,
    route_late_count,
    ROUND(route_late_count * 100.0 / NULLIF(route_order_volume, 0), 2) AS route_late_rate_pct,
    ROUND(avg_transit_days, 1) AS mean_transit_days,
    ROUND(avg_freight_cost, 2) AS mean_freight_brl,
    CASE 
        WHEN route_late_count * 100.0 / NULLIF(route_order_volume, 0) > 15.0 THEN 'High Bottleneck Risk'
        WHEN route_late_count * 100.0 / NULLIF(route_order_volume, 0) > 8.0 THEN 'Moderate Risk'
        ELSE 'SLA Compliant Corridor'
    END AS route_health_status
FROM route_stats
WHERE route_order_volume >= 50
ORDER BY route_late_rate_pct DESC;
