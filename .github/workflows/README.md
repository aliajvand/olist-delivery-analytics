# Workflows

`tests.yml` runs on every push and pull request. It needs no secrets and no
dataset: `scripts/make_sample_data.py` generates Olist-shaped synthetic rows and
`scripts/run_smoke_test.py` pushes them through cleaning, features and the model
comparison.

_Part of **Olist Delivery Performance Analytics**. Project overview: [`../../README.md`](../../README.md)._
