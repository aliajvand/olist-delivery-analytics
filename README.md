# Olist delivery performance: from raw CSVs to deployable decision policies

End-to-end analytics and ML project on the [Olist Brazilian E-Commerce public
dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): ~99k orders,
~3k sellers, 27 Brazilian states, Sep 2016 - Oct 2018.

**The question:** which orders will be delivered late, what date should we promise,
and what will the freight cost - all answered using only what is known the moment
the customer clicks buy.

**Business framing.** Decision owner: logistics operations. Decision: which orders
to expedite, and what delivery date to display at checkout. Unit of decision: one
order, at `Order_Purchase_Timestamp`.

---

## Headline findings

1. **The delivery promise is padded but under-indexed on distance.** The median order
   arrives days *early*, yet the late rate on 1000km+ shipments is several times the
   sub-50km rate, because the promise barely scales with distance while actual transit
   does. **Scaling the promised date with distance is a SQL change and should ship
   before any model.**
2. **Late delivery is predictable at purchase time.** A chronologically validated
   classifier concentrates a multiple-x lift over random selection into a fixed daily
   expedite capacity. The strongest driver is the promise itself, then distance, then
   leakage-free seller reliability.
3. **Delivery duration is predictable enough to tighten the promise.** Converted into
   a quantile promise policy, the model holds the incumbent's service level on a
   measurably tighter promise - a checkout conversion lever, not an MAE number.
4. **Freight is highly predictable** and lands within a fraction of a percent of
   actual total freight over a month, so it is usable for accrual today.
5. **Delivery timing dominates customer satisfaction.** Late orders draw
   substantially worse review scores, and the decay is monotonic in delay size.
   That is the business case for spending expedite budget at all.
6. **Revenue concentration is extreme on both sides** - a small fraction of sellers
   and three categories carry most of GMV, so seller churn is a revenue event.

Every figure carries an insight computed from the data at runtime, so refreshing the
dataset can never leave a stale claim on a chart.

---

## Verification status

The pipeline, the leakage guarantees and the modeling contract are verified without the
120 MB Kaggle download: `scripts/make_sample_data.py` generates Olist-SHAPED synthetic
rows and `scripts/run_smoke_test.py` pushes them through cleaning, features and the model
comparison.

| Check | Result |
|---|---|
| Test suite (`tests/`) | **34 passed, 0 failed** |
| Declared-grain audit | 8/8 tables PASS |
| Money and count reconciliation | 4/4 PASS |
| Leakage audit on the purchase-time contract | 39/39 features allowed |
| Model base built | 8,000 x 48, chronological train/valid/test split |
| Headless pipeline (`scripts/run_pipeline.py`) | green, 8.5s on synthetic data |

Report: [`reports/pipeline_smoke_test.md`](reports/pipeline_smoke_test.md).

**The numbers in that report describe the pipeline, not the Brazilian market.** Synthetic
rows can never be quoted as findings. The headline findings above come from the real
dataset and are reproduced by `make pipeline` once `data/raw/` holds the nine Olist CSVs.

---

## What makes this more than a notebook dump

| | |
|---|---|
| **Leakage-safe history** | Seller, route and customer reliability are built with `merge_asof` on *delivery* time and joined on *purchase* time, so an order can only see outcomes that had already happened. The naive `groupby().mean()` version leaks the future. Three proofs in notebook 04, four regression tests. |
| **Chronological validation** | Train / valid / test split by purchase date, never random. Over 24 months volume grows ~20x and the seller mix turns over; a random split would train on August to predict March. |
| **Real baselines** | Not just `DummyClassifier`. Projects are benchmarked against the actual incumbent: Olist's own promise date, a BRL/kg rate card, a distance rule. Beating a dummy proves nothing. |
| **Decision policies, not metrics** | Fixed expedite capacity (stable under score drift, unlike a threshold), a quantile promise with a service-level dial, service-tier routing. Each with a cost-sensitivity table because the cost inputs are assumptions, not data. |
| **Calibration** | Probabilities are isotonic-calibrated on validation before being multiplied by money. An uncalibrated score is a ranking, not a probability. |
| **Tested library + thin notebooks** | Logic lives in `src/olist/` as pure functions with 34 tests. Notebooks tell the story; `scripts/run_pipeline.py` rebuilds the data headlessly. Reproducing this never requires opening Jupyter. |
| **Stated limits** | No profit or ROI figure anywhere - the dataset has no COGS, commission or marketing spend. No causal claims. Every model ships with a monitoring table naming what will break it. |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Put the nine Olist CSVs in data/raw/  (see data/raw/README.md)
# 2. Build the clean + feature layers
python scripts/run_pipeline.py

# 3. Verify the leakage guarantees and the cleaning contract
pytest tests -q

# 4. Run the notebooks in order
jupyter lab
```

No Olist download yet? Generate a synthetic, Olist-*shaped* sample and smoke-test
the whole pipeline:

```bash
python scripts/make_sample_data.py --out data/sample_raw --orders 8000
python scripts/run_pipeline.py --raw-dir data/sample_raw --out-dir data/sample_processed
```

The sample is for testing only. Never report a number from it.

---

## Notebook order

| Notebook | Phase | What it does |
|---|---|---|
| `01_scope_and_audit.ipynb` | 0 - Scope | Decision, grain, provenance. Raw schema, null and grain audit. Finds the two grain violations that phase 1 repairs. |
| `02_data_cleaning.ipynb` | 1 - Clean | Null decisions with indicators, dedup, timestamp monotonicity audit, referential integrity, reconciliation gate. |
| `04_feature_engineering.ipynb` | 2 - Features | Grain declarations, geo distance, targets, **leakage-safe as-of history**, the ML base table, generated feature catalogue. |
| `03_eda.ipynb` | 3 - EDA/BI | 15 business questions, 16 figures, Python validation aggregates for BI reconciliation. |
| `05_modeling.ipynb` | 4 - Modeling | The shared protocol and index. Also documents the three targets that were **rejected** and why. |
| `modeling/01..05` | 4 - Modeling | Five projects: binary classification, two regressions, multi-class, clustering. |
| `06_evaluation.ipynb` | 4 - Modeling | Reads every `model_metadata.json` back into one scorecard, plus a reproducibility audit and the portfolio read-out. |

Run 01 -> 02 -> 04 -> 03 -> 05 -> modeling/01-05 -> 06. Notebook 04 comes before 03
because the EDA consumes the feature tables.

## The five modelling projects

| # | Type | Target | Baseline it must beat | Deliverable |
|---|---|---|---|---|
| 01 | Binary classification | `late_delivery_flag` | distance + tight-promise rules | flag the riskiest N% at fixed expedite capacity |
| 02 | Regression | `delivery_days` | **Olist's own promise date** | quantile promise policy with a service-level dial |
| 03 | Regression | `order_freight_total` | training-period BRL/kg rate card | portfolio accrual + p90 quote buffer |
| 04 | Multi-class | `outcome_band` (5 tiers) | always predict the largest band | service-tier routing, guarded on non-adjacent errors |
| 05 | Clustering | none | n/a | 4 fulfilment playbooks, validated on a held-out outcome the clustering never saw |

Project 04 originally targeted `Seller_State`. That was dropped: coordinates
determine the state, so the target is a deterministic function of its own features
and scores ~100% while meaning nothing. Recognising that is the point.

---

## Layout

```
├── README.md · ROADMAP.md · requirements.txt · Makefile
├── data/
│   ├── raw/                    the nine Olist CSVs (gitignored, read-only)
│   └── processed/              cleaning/ · features/ · ml/   (generated)
├── src/olist/
│   ├── config.py               paths, cutoff dates, the banned-column contract
│   ├── io_utils.py             read raw, read/write parquet, table summaries
│   ├── cleaning.py             phase 1 as pure functions
│   ├── features.py             phase 2, including the as-of leakage-safe joins
│   ├── modeling.py             split, baselines, candidates, policies, artifacts
│   └── plots.py                shared figure helpers with runtime insights
├── notebooks/                  the narrative (+ back-compat shims)
├── scripts/
│   ├── run_pipeline.py         rebuild clean + features headlessly, with gate checks
│   └── make_sample_data.py     synthetic Olist-shaped data for CI and smoke tests
├── tests/                      34 tests: cleaning contract, leakage guarantees, policies
├── models/<project>/           best_model.joblib + model_metadata.json  (generated)
└── reports/
    ├── figures/                qNN_*.png + model/<project>/*.png        (generated)
    ├── feature_catalogue.md    generated by notebook 04, never hand-edited
    ├── bi_semantic_model.md    star schema and measure definitions
    └── bi_validation_aggregates.csv   Python truth for BI reconciliation
```

## Reproducibility

Fixed seed (42) and a **fixed snapshot date** (2018-10-18) everywhere -
`pd.Timestamp.now()` appears nowhere, so recency and as-of features do not silently
change depending on when the code is run. Every trained model writes its feature
list, metrics, baseline, decision policy, seed, snapshot and split boundaries to
`model_metadata.json`; notebook 06 audits that all of it is present.

## Known limitations

* **No profit.** No COGS, commission or marketing spend in the dataset. Revenue is
  labelled "GMV proxy" throughout. Money figures in phase 4 rest on stated
  assumptions and ship with sensitivity tables.
* **No causality.** Observational data. The promise being the top predictor of
  lateness does not mean shortening it reduces delay.
* **No carrier identity** - probably the largest missing variable, and a hard
  ceiling on accuracy for projects 01, 02 and 04.
* **Zip-prefix centroid distances**, so intra-city distance is noisy.
* **Undelivered orders are dropped**, which mildly under-samples the worst outcomes.
* **One short test window**, no bootstrap confidence intervals yet.
* 2016-2018 Brazilian marketplace logistics. Do not transfer.

## Licence

Code: MIT. Data: Olist dataset, CC BY-NC-SA 4.0, not redistributed here.
