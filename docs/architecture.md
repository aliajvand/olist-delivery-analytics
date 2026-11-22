# System Architecture: Olist Delivery Analytics

```mermaid
flowchart TD
    subgraph Ingestion_Layer ["E-Commerce Relational Data Layer"]
        OlistRaw["Olist Public Brazilian E-Commerce<br/>(100K+ Orders | Sellers | Geolocation)"] --> StarSchema[("Star Schema Storage<br/>fact_orders | dim_sellers | dim_customers")]
    end

    subgraph Feature_Pipeline ["Zero-Carrier-Scan Feature Engineering"]
        StarSchema --> PreOrder["39 Purchase-Time Features Only<br/>(Order GMV, Weight, Distance, Historical Priors)"]
        PreOrder --> Filter["Strict Causal Event Masking<br/>(Zero Carrier Scan or Transit Milestone Leakage)"]
    end

    subgraph Modeling_Engine ["Fulfillment Decision Engine"]
        Filter --> ChronoSplit["Chronological Split<br/>(Train < May 2018 | Valid < Jul 2018)"]
        ChronoSplit --> Classifier["Late Delivery Risk Classifier<br/>(Logistic Regression & HistGBM)"]
        ChronoSplit --> Regressor["Transit Lead Time Regressor<br/>(MAE: 4.09 Days)"]
        Classifier --> Policy["Fixed-Capacity Expedite Policy<br/>(Top 5% Queue Capacity -> 2.07x Lift)"]
    end

    subgraph Serving_Layer ["Operational Logistics Deployment"]
        Policy & Regressor --> FastApi["FastAPI Backend<br/>(/predict-delay, /route-risk, /governance)"]
        FastApi --> DashPortal["Plotly/Dash Fulfillment Cockpit<br/>(Interactive Order Simulator & Corridor Analytics)"]
        FastApi --> SQLAudit["Star Schema SQL Analytics<br/>(Seller SLA Pareto & Route Bottlenecks)"]
    end
```

## Architectural Tenets
1. **Zero Post-Purchase Leakage:** Every feature strictly precedes or coincides with `order_purchase_timestamp`.
2. **Capacity-Constrained Policy:** Dispatch decisions reflect real warehouse expedite bandwidth rather than abstract probability thresholds.
3. **End-to-End Governance:** Chronological splitting mirrors production deployment reality.

