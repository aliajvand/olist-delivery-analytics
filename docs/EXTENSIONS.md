# Architectural & Modeling Extensions: Olist Delivery Analytics

**Author:** Ali Ajvand  
**Scope:** Advanced capabilities aligned with verified resume competencies (Classification, Operational Policy Design, Logistics Analytics, SQL, FastAPI, Plotly/Dash).

---

## 1. Conformal Prediction for Customer Dynamic Promise Dates

### 1.1 Business Motivation
Static promise dates (e.g., "Expected in 15 days") lead to either excessive customer churn or premature late delivery penalties. Conformal prediction guarantees finite-sample coverage at user-specified confidence $1 - \alpha$.

### 1.2 Mathematical Formulation
Using non-conformity scores on calibration fold $S_{\text{cal}}$:
$$R_i = |y_i - \hat{\mu}(x_i)|$$
For target coverage $1 - \alpha = 90\%$, let $\hat{q}$ be the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of $\{R_i\}$.
The valid prediction interval is:
$$C(x) = [\hat{\mu}(x) - \hat{q}, \; \hat{\mu}(x) + \hat{q}]$$
Customer checkout displays: *"Guaranteed delivery by [Date] with 90% statistical certainty"*.

---

## 2. Dynamic Carrier Re-Assignment & Multi-Carrier Routing

### 2.1 Logistics Architecture
- When `predict-delay` flags an order exceeding the 25.1% threshold, automatically trigger a secondary routing query.
- Shift origin fulfillment from regional postal services (Correios standard ground) to regional private courier express lines, recalculating expected transit time and net margin.
