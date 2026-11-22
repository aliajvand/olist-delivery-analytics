"""Split integrity, leakage guard and the decision policies."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from olist import config as cfg, features, modeling as ml


def test_leakage_audit_blocks_the_target():
    with pytest.raises(ValueError, match="Leakage audit failed"):
        ml.leakage_audit(["order_gmv", "late_delivery_flag"], target="late_delivery_flag")


def test_leakage_audit_blocks_identifiers_and_post_outcome():
    with pytest.raises(ValueError):
        ml.leakage_audit(["Order_Id"], target="late_delivery_flag")
    with pytest.raises(ValueError):
        ml.leakage_audit(["delivery_delay_days"], target="late_delivery_flag")


def test_leakage_audit_passes_the_real_contract():
    audit = ml.leakage_audit(features.PURCHASE_TIME_FEATURES, target="late_delivery_flag")
    assert (audit["verdict"] == "allowed").all()


def test_chronological_split_drops_rows_without_a_target(feature_tables):
    base = feature_tables["order_model_base"]
    cols = [c for c in features.PURCHASE_TIME_FEATURES if c in base.columns]
    data = ml.chronological_split(base, cols, "late_delivery_flag")
    total = len(data.X_train) + len(data.X_valid) + len(data.X_test)
    assert total == int(base["late_delivery_flag"].notna().sum())
    assert data.y_train.notna().all()


def test_chronological_split_raises_on_unknown_feature(feature_tables):
    with pytest.raises(KeyError):
        ml.chronological_split(
            feature_tables["order_model_base"], ["not_a_column"], "late_delivery_flag"
        )


def test_preprocessor_is_fitted_on_train_only(feature_tables):
    """Unseen test categories must not crash or silently create new columns."""
    base = feature_tables["order_model_base"]
    cols = [c for c in features.PURCHASE_TIME_FEATURES if c in base.columns]
    data = ml.chronological_split(base, cols, "late_delivery_flag")
    pipe = ml.make_pipeline(LogisticRegression(max_iter=500), ml.to_sklearn_frame(data.X_train))
    pipe.fit(ml.to_sklearn_frame(data.X_train), data.y_train.astype(int))
    n_train_cols = pipe.named_steps["pre"].transform(ml.to_sklearn_frame(data.X_train)).shape[1]
    n_test_cols = pipe.named_steps["pre"].transform(ml.to_sklearn_frame(data.X_test)).shape[1]
    assert n_train_cols == n_test_cols


def test_baseline_is_beaten_or_reported(feature_tables):
    base = feature_tables["order_model_base"]
    cols = [c for c in features.PURCHASE_TIME_FEATURES if c in base.columns]
    data = ml.chronological_split(base, cols, "late_delivery_flag")
    candidates = {
        **ml.baseline_classifiers(data.y_train),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }
    table, fitted = ml.compare_candidates(candidates, data, task="classification")
    assert {"baseline_majority", "baseline_prior", "logreg"} == set(table["model"])
    assert "pr_auc" in table.columns
    assert set(fitted) == set(candidates)


def test_threshold_policy_finds_the_minimum_cost():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, 4000)
    p = np.clip(y * 0.45 + rng.beta(2, 6, 4000), 0, 1)
    table = ml.threshold_policy(y, p, cost_fp=1.0, cost_fn=10.0)
    assert table["is_min_cost"].sum() >= 1
    best = table.loc[table["is_min_cost"]].iloc[0]
    assert best["expected_cost"] == table["expected_cost"].min()
    assert table["recall"].between(0, 1).all()


def test_capacity_policy_beats_random():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, 5000)
    p = np.clip(y * 0.5 + rng.beta(2, 5, 5000), 0, 1)
    result = ml.capacity_policy(y, p, capacity_pct=10)
    assert result["orders_flagged"] == 500
    assert result["lift_vs_random"] > 1
    assert 0 <= result["recall_at_capacity"] <= 1


def test_calibration_table_shape():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.3, 2000)
    p = rng.uniform(size=2000)
    calib = ml.calibration_table(y, p, bins=10)
    assert len(calib) == 10
    assert calib["orders"].sum() == 2000


def test_regression_metrics_are_sane():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = ml.regression_metrics(y, y)
    assert m["mae"] == 0
    assert m["r2"] == 1
