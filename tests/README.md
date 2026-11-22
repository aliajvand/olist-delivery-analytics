# Tests

```bash
pytest
```

The entire suite runs on a fresh clone with **no data download**: the synthetic generator in `scripts/make_sample_data.py` stands in for the real source.

What is actually tested:

- **Repository hygiene**: no absolute machine paths, no stored tracebacks in notebooks, no unfinished docs, no broken relative links, a README in every folder.
- **The leakage contract**: the columns this project bans are provably absent from the stored feature lists.
- **Published numbers**: the metrics in the governance summary reconcile with each other, so a claim in the README cannot quietly rot.
- **Helper regressions**: each of these covers a bug that was real once.

_Part of **Olist Delivery Performance Analytics**. Project overview: [`../README.md`](../README.md)._
