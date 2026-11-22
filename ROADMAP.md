# Roadmap

Status of every phase. Each item links to the notebook that owns it.

## Phase 0: Scope - DONE (`notebooks/01_scope_and_audit.ipynb`)
- [x] Confirm decision owner, target and grain - operations lead; `late_delivery_flag`; one order at purchase time
- [x] Download one documented dataset version - single Kaggle release, nine CSVs
- [x] Record source, citation, retrieval date, rows and columns - provenance dict + row-count gate vs the canonical release

## Phase 1: Clean - DONE (`notebooks/02_data_cleaning.ipynb`, `src/olist/cleaning.py`)
- [x] Load without modifying raw files - `io.load_raw` is read-only
- [x] Audit schema, nulls, duplicates, ranges, relationships - per-column null decisions, timestamp monotonicity audit, referential integrity (0 orphans)
- [x] Reconcile counts and totals - money drift 0.00%, asserted in-notebook
- [x] Export clean tables - nine parquet files in `data/processed/cleaning/`

Repairs: reviews deduplicated to one per order, geolocation collapsed to one row
per zip prefix, coordinates outside Brazil rejected. Outliers flagged, never dropped.

## Phase 2: Features - DONE (`notebooks/04_feature_engineering.ipynb`, `src/olist/features.py`)
- [x] Declare feature grain - eight tables, one declared key each, asserted by `grain_audit`
- [x] Define observation window and horizon - decision point = purchase timestamp, horizon 1-60 days
- [x] Build leakage-safe features - as-of `merge_asof` seller / route / customer history, shrunk to a global prior. Three in-notebook proofs, four regression tests
- [x] Write the feature catalogue - `reports/feature_catalogue.md`, generated from code so it cannot drift

## Phase 3: EDA and BI - DONE (`notebooks/03_eda.ipynb`)
- [x] Answer the business questions - 15 questions, 16 figures, each with a runtime-computed insight
- [x] Build Python validation aggregates - `reports/bi_validation_aggregates.csv`
- [x] Build the semantic model and report - star schema and measure definitions in `reports/bi_semantic_model.md` (the .pbix itself is not in this repo)
- [x] Reconcile BI measures with Python - reconciliation gate passes at < 0.01% drift; fill the `bi_value` column once the report is built

## Phase 4: Modeling - DONE (`notebooks/05_modeling.ipynb`, `notebooks/modeling/01-05`, `notebooks/06_evaluation.ipynb`)
- [x] Naive or rules baseline - every project is benchmarked against the real incumbent, not a dummy
- [x] Valid split strategy - chronological train/valid/test, one shared `split` column, non-overlap asserted
- [x] Candidate comparison - 4-5 candidates per project, scored on validation; the test set is opened once
- [x] Calibration, explanation, threshold or horizon policy - isotonic calibration, permutation importance on held-out data, capacity / quantile / tier policies with cost sensitivity
- [x] Limitations and monitoring notes - every project ends with a limitations list and a monitoring table with triggers

## Backlog, in priority order
1. Ship the distance-scaled promise date. No model needed, largest single effect.
2. Source carrier or shipping-method identity - the biggest missing variable.
3. Model project 04 as a proper ordinal target (cumulative logit) instead of independent classes.
4. Bootstrap confidence intervals on the test metrics; one short test window has no error bars.
5. Rolling-origin backtest for project 01 instead of a single holdout period.
6. Build the actual Power BI report against `reports/bi_semantic_model.md` and fill in the reconciliation column.
