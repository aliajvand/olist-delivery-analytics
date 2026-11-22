import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import build  # noqa: E402
from olist import cleaning, features  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    """A small Olist-shaped synthetic dataset. Fast enough to rebuild per session."""
    return build(n_orders=1200, seed=7)


@pytest.fixture(scope="session")
def clean_and_audits(raw):
    """Build clean tables once and share both tables and audits."""
    return cleaning.build_clean_tables(raw)


@pytest.fixture(scope="session")
def clean(clean_and_audits):
    tables, _ = clean_and_audits
    return tables


@pytest.fixture(scope="session")
def clean_audits(clean_and_audits):
    _, audits = clean_and_audits
    return audits


@pytest.fixture(scope="session")
def feature_tables(clean):
    tables, _ = features.build_feature_tables(clean)
    return tables
