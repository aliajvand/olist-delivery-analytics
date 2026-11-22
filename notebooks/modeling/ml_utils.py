"""Backwards-compatible shim over `olist.modeling` and `olist.plots`.

Kept so existing `import ml_utils as mlu` cells keep running. New code should
`from olist import modeling as ml, plots`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "olist").exists())
sys.path.insert(0, str(ROOT / "src"))

from olist.modeling import (  # noqa: F401
    baseline_classifiers, baseline_regressors, calibrate_prefit, calibration_table,
    capacity_policy, chronological_split, classification_metrics, cluster_scan,
    coefficient_table, compare_candidates, importance_table, leakage_audit,
    make_pipeline, multiclass_metrics, regression_metrics, threshold_policy,
    to_sklearn_frame,
)
from olist.plots import (  # noqa: F401
    plot_calibration, plot_classification_report, plot_cost_curve,
    plot_regression_diagnostics, stamp_and_save,
)

# Legacy aliases
evaluate_model = classification_metrics
get_feature_importance = importance_table
