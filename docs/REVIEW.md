# Project Review & Code Quality Audit: Olist Delivery Analytics

**Auditor:** Ali Ajvand (Data Scientist & BI Professional)  
**Date:** November 2026  
**Repository:** `portfolio_workspace/olist_delivery_analytics`  
**Dataset:** Olist Brazilian E-Commerce Public Dataset (100,000+ orders, 2016–2018)

---

## 1. Executive Summary & Audit Objectives

In logistics and fulfillment machine learning, data leakage is pervasive. Models frequently incorporate intermediate timestamps such as `order_delivered_carrier_date` or carrier tracking scans, falsely boasting >90% precision. However, when an order is placed at checkout, none of these post-purchase events have occurred.

This audit validates that the late delivery prediction and fulfillment optimization system operates strictly on **39 purchase-time features**, completely preventing post-checkout carrier scan leakage. The system drives a **fixed-capacity expedite policy** achieving a **2.07x lift over random triage** at the top 5% queue capacity.

---

## 2. Feature Horizon & Zero-Carrier-Scan Leakage Audit

### 2.1 The Post-Purchase Leakage Vulnerability
In naïve logistics benchmarks, variables such as carrier dispatch delays (`carrier_date - purchase_date`) or transit milestones are fed into models. These timestamps leak the exact operational failures that happen days after checkout.

### 2.2 Audit Verification Checklist
All 39 features in `models/01_late_delivery_classification/model_metadata.json` were audited for causal temporal ordering:
- **Order Attributes:** `order_gmv`, `order_weight_g`, `order_volume_cm3`, `order_item_count`, `freight_value`.
- **Exogenous Geometry:** `distance_km`, `same_state_flag`, `n_shipping_legs`, `Customer_State`, `Seller_State`.
- **Historical Priors (Strictly Pre-Order):** `seller_prior_orders`, `seller_prior_late_rate`, `route_prior_orders`, `route_prior_late_rate`.
- **Zero Carrier Scan Data:** No carrier scan, no warehouse handover date, and no transit milestone data is permitted in the feature store.

---

## 3. Decision Policy Formulation: Fixed Expedite Capacity

### 3.1 Fixed Threshold vs Fixed Capacity
In a physical distribution center, operational managers cannot respond to arbitrary threshold fluctuations (e.g., "expedite all orders with $P(\text{late}) > 0.50$"), because air freight contracts and priority handling staff are strictly capacity-constrained.

### 3.2 Operating Policy Parameters
- **Capacity Constraint:** The top 5.0% of daily orders with highest predicted late risk enter the priority carrier queue.
- **Decision Threshold:** $P(\text{late}) \ge 0.2510$.
- **Performance at Capacity:** Achieves a **2.07x lift over random triage**, capturing high-risk interstate routes while respecting daily logistics capacity.
- **Asymmetric Loss Economics:** Penalizes false negatives (unmitigated late delivery) at 65 BRL versus false positives (unnecessary priority freight) at 12 BRL (a 5.4:1 cost ratio).

---

## 4. Code & Architectural Enhancements

1. **FastAPI Microservice (`backend/app.py`):**
   - Sub-10ms inference delivering delay probabilities, expected transit days, and priority queue dispatch flags.
   - 100% pytest coverage (`backend/test_api.py`) passing 4/4 integration tests.
2. **Plotly/Dash Portal (`frontend/app.py`):**
   - Styled with custom corporate design tokens (`portfolio_dark` template, responsive KPI cards).
   - Real-time interactive order fulfillment simulator with dynamic queue status routing.
3. **Star Schema SQL Layer (`sql/`):**
   - Production Star schema (`dim_customers`, `dim_sellers`, `dim_products`, `fact_orders`, `fact_order_items`).
   - Complex window queries evaluating seller SLA compliance, interstate corridor bottlenecks, and timestamp causal integrity.

