# Data

Everything the pipeline reads or writes. **Nothing in `raw/` or `external/` is committed**: the licence forbids it, the size makes it silly, or both. Download instructions are in [`raw/README.md`](raw/README.md), and `make sample` generates an Olist-shaped synthetic stand-in so the pipeline runs without it.

| item | what it is |
|---|---|
| `raw/` | the untouched source download. Read-only. Never edited in place, never committed. |
| `external/` | supporting reference tables from a different source than the primary one. |
| `processed/cleaning/` | validated, renamed, correctly typed tables. One row per source row unless an exclusion is documented. |
| `processed/features/` | model-ready feature tables with a documented grain. |
| `processed/ml/` | the exact train and test splits, so a metric can be reproduced rather than re-rolled. |
| `sample/` | synthetic stand-in data from `scripts/make_sample_data.py`. Correct shape, invented values. Never quote a number computed from it. |

_Part of **Olist Delivery Performance Analytics**. Project overview: [`../README.md`](../README.md)._
