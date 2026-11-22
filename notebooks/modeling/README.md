# Modelling notebooks

One notebook per modelling project, each with the same spine: baseline first, leakage audit, validation that matches how the data was generated, one defensible model, explainability, then a business read-out.

`ml_config.py` holds the shared paths and the seed. `ml_utils.py` and `ml_utils_extra.py` hold the evaluation and plotting helpers, so no notebook re-implements a metric.

_Part of **Olist Delivery Performance Analytics**. Project overview: [`../../README.md`](../../README.md)._
