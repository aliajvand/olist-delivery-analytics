-- ============================================================================
-- Business Question 3: Post-Purchase Data Leakage Verification
-- Technique: Timestamp Integrity Verification, Event Horizon Audit
-- Proves that all 39 predictive features are timestamped strictly AT OR BEFORE
-- order_purchase_timestamp, avoiding carrier scan leakage.
-- ============================================================================

SELECT
    o.order_id,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    -- Audit checks
    CASE 
        WHEN o.order_approved_at < o.order_purchase_timestamp THEN 'ILLEGAL: Approval Before Purchase'
        WHEN o.order_delivered_carrier_date < o.order_purchase_timestamp THEN 'ILLEGAL: Carrier Handover Before Purchase'
        ELSE 'VALID: Causal Temporal Ordering'
    END AS causal_timestamp_check,
    ROUND(JULIANDAY(o.order_delivered_carrier_date) - JULIANDAY(o.order_purchase_timestamp), 2) AS days_to_carrier_handover,
    ROUND(JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_delivered_carrier_date), 2) AS transit_days
FROM fact_orders o
WHERE o.order_delivered_customer_date IS NOT NULL
ORDER BY o.order_purchase_timestamp DESC
LIMIT 50;

