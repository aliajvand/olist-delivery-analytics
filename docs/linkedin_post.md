# LinkedIn Showcase Post: Solving E-Commerce Late Deliveries Without Cheating with Data Leakage

In logistics data science, it is astonishingly easy to build a model with 95% accuracy if you cheat without knowing it.

If your model uses features like `carrier_handover_timestamp`, warehouse scan milestones, or package weigh-in times, you aren't forecasting delays—you are describing events that already occurred days after the customer clicked "Place Order."

I built and audited an enterprise fulfillment intelligence and expedite dispatch platform evaluated on the Olist Brazilian E-Commerce dataset (100K+ orders across 3,000+ sellers).

### Key Engineering Decisions:
1. **Strict 39 Purchase-Time Feature Boundary:** Every predictive variable is derived strictly at checkout (package dimensions, GMV, freight ratio, geodesic distance, and pre-purchase seller historical performance). Zero carrier tracking scan data is permitted.
2. **Fixed-Capacity Expedite Policy (2.07x Lift):** In real warehouses, air freight and priority handling have strict physical budget caps. Rather than choosing arbitrary probability thresholds, the system ranks orders by risk and routes the **top 5.0% highest-risk orders** into priority carrier lanes, achieving a **2.07x lift over random triage**.
3. **Asymmetric Loss Economics:** Penalizes unmitigated late delivery (65 BRL in customer compensation and churn risk) against the cost of priority shipping (12 BRL), achieving an optimal 5.4:1 cost ratio.
4. **Production Architecture:**
   - **FastAPI Backend:** Sub-10ms inference serving real-time delay probabilities, expected transit lead times, and dispatch recommendations with 100% test coverage.
   - **Plotly/Dash Portal:** Modern dark-theme fulfillment cockpit enabling operations managers to simulate live checkout risk and inspect interstate bottleneck corridors.
   - **Analytical SQL Layer:** Relational Star schema tracking seller SLA fulfillment Pareto curves and transit lead time distributions.

The complete code, validation ledgers, and interactive portal are open-source in the repository below.

#DataScience #SupplyChain #LogisticsAnalytics #MachineLearning #FastAPI #PlotlyDash #Python #DataLeakage

