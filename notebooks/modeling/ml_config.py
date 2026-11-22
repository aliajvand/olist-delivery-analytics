"""Backwards-compatible shim over `olist.config` + `olist.modeling`.

Kept so existing `import ml_config as cfg` cells keep running. New code should
`from olist import config as cfg`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "olist").exists())
sys.path.insert(0, str(ROOT / "src"))

from olist.config import (  # noqa: F401
    CLEAN_DIR, FEATURES_DIR, FIGURES_DIR, ML_DIR, MODELS_DIR, MODEL_FIGURES_DIR,
    N_JOBS, RANDOM_STATE, SNAPSHOT_DATE, TRAIN_END, VALID_END, CV_FOLDS,
    DATASET_SOURCE, ensure_dirs, set_seed,
)
from olist.io_utils import load_clean, load_features, load_ml, save_json  # noqa: F401
from olist.modeling import (  # noqa: F401
    ProjectPaths, get_paths, save_artifacts, to_sklearn_frame,
)

TEST_SIZE = 0.20  # unused: this project splits chronologically, not randomly


def save_model(model, paths, filename="best_model.joblib"):
    import joblib
    path = paths.model(filename)
    joblib.dump(model, path)
    print("saved model ->", path)
    return path
