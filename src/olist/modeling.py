"""Phase 4 - Modeling.

The pieces a reviewer looks for, all in one reusable place:
  * leakage_audit          - proves no post-outcome column reached the matrix
  * chronological_split    - split that matches how the model would be used
  * build_preprocessor     - fitted on train only, inside a Pipeline
  * compare_candidates     - baseline first, then real models, same protocol
  * threshold_policy       - turns probabilities into a decision with a cost model
  * capacity_policy        - the version ops actually uses: fixed daily capacity
  * calibration_table      - are the probabilities usable as probabilities?
  * ModelArtifacts.save    - model + metrics + feature list + metadata to disk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from . import io_utils as io

CATEGORICAL_DTYPES = ("object", "string", "category")


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
def leakage_audit(features: Sequence[str], target: str) -> pd.DataFrame:
    """Check the feature list against the project's banned-column rules.

    Run this in every modelling notebook and print it. It is cheap, and it is
    the single most common thing that invalidates a portfolio project.
    """
    rows = []
    for name in features:
        reasons = []
        if name == target:
            reasons.append("is the target")
        if name.lower().endswith(cfg.IDENTIFIER_PATTERNS):
            reasons.append("identifier")
        if name in cfg.POST_OUTCOME_COLUMNS:
            reasons.append("known only after the outcome")
        rows.append(
            {
                "feature": name,
                "verdict": "BLOCK" if reasons else "allowed",
                "reason": "; ".join(reasons) or "known at purchase time",
            }
        )
    audit = pd.DataFrame(rows)
    blocked = audit.loc[audit["verdict"] == "BLOCK", "feature"].tolist()
    if blocked:
        raise ValueError(f"Leakage audit failed for target '{target}'. Remove: {blocked}")
    return audit


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, X, y in (
            ("train", self.X_train, self.y_train),
            ("valid", self.X_valid, self.y_valid),
            ("test", self.X_test, self.y_test),
        ):
            rows.append(
                {
                    "split": name,
                    "rows": len(X),
                    "share_pct": round(len(X) / max(len(self.X_train) + len(self.X_valid) + len(self.X_test), 1) * 100, 1),
                    "target_mean": round(float(pd.to_numeric(y, errors="coerce").mean()), 4),
                }
            )
        return pd.DataFrame(rows)


def chronological_split(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    time_col: str = "Order_Purchase_Timestamp",
    split_col: str = "split",
) -> SplitData:
    """Split on the pre-computed `split` column, dropping rows with no target.

    A random split on this dataset is optimistic: seller behaviour, carrier
    capacity and the delivery promise all drift over the 24 months of data. The
    honest question is "does a model trained on the past work on the future".
    """
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    before = len(work)
    work = work.loc[work[target].notna()]
    dropped = before - len(work)

    missing = [c for c in features if c not in work.columns]
    if missing:
        raise KeyError(f"Features missing from the table: {missing}")

    parts = {}
    for name in ("train", "valid", "test"):
        chunk = work.loc[work[split_col] == name]
        parts[name] = (chunk[list(features)].copy(), chunk[target].copy())

    meta = pd.DataFrame(
        [
            {
                "rows_dropped_no_target": dropped,
                "train_period": _period(work, split_col, "train", time_col),
                "valid_period": _period(work, split_col, "valid", time_col),
                "test_period": _period(work, split_col, "test", time_col),
            }
        ]
    )
    return SplitData(
        X_train=parts["train"][0],
        X_valid=parts["valid"][0],
        X_test=parts["test"][0],
        y_train=parts["train"][1],
        y_valid=parts["valid"][1],
        y_test=parts["test"][1],
        meta=meta,
    )


def _period(df: pd.DataFrame, split_col: str, name: str, time_col: str) -> str:
    chunk = df.loc[df[split_col] == name, time_col]
    if chunk.empty:
        return "empty"
    return f"{chunk.min():%Y-%m-%d} .. {chunk.max():%Y-%m-%d}"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def split_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(include=list(CATEGORICAL_DTYPES)).columns.tolist()
    return numeric, categorical


def to_sklearn_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas nullable dtypes to plain float/object so sklearn is happy."""
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) in {"Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "boolean"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif str(out[col].dtype) in {"string", "category"}:
            out[col] = out[col].astype("object").where(out[col].notna(), np.nan)
    return out


def build_preprocessor(X: pd.DataFrame, scale: bool = True, min_frequency: int = 20) -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot categoricals.

    `min_frequency` folds rare categories into `infrequent_sklearn`, which keeps
    the matrix narrow and stops the model memorising one-off city names.
    """
    numeric, categorical = split_column_types(X)
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=min_frequency,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator, X: pd.DataFrame, scale: bool = True) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(X, scale=scale)), ("model", estimator)])


def feature_names_out(pipe: Pipeline) -> List[str]:
    try:
        return list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        return []


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_proba)
        m["pr_auc"] = average_precision_score(y_true, y_proba)
        m["brier"] = brier_score_loss(y_true, y_proba)
    return {k: float(v) for k, v in m.items()}


def multiclass_metrics(y_true, y_pred, y_proba=None, labels=None) -> Dict[str, float]:
    """Macro averaging, because the interesting classes are the rare ones."""
    from sklearn.metrics import cohen_kappa_score

    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            m["macro_roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                               average="macro", labels=labels)
        except Exception:
            pass
    return {k: float(v) for k, v in m.items()}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan"),
        "median_abs_err": float(np.median(np.abs(y_true - y_pred))),
    }


# --------------------------------------------------------------------------
# Candidate comparison
# --------------------------------------------------------------------------
def compare_candidates(
    candidates: Dict[str, object],
    data: SplitData,
    task: str,
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Fit every candidate on train, score on validation, same preprocessing.

    The test set is not touched here. It is spent once, on the single model
    chosen from this table.
    """
    if task not in {"classification", "multiclass", "regression"}:
        raise ValueError("task must be 'classification', 'multiclass' or 'regression'")

    X_train = to_sklearn_frame(data.X_train)
    X_valid = to_sklearn_frame(data.X_valid)
    y_train, y_valid = data.y_train, data.y_valid
    if task in {"classification", "multiclass"}:
        y_train, y_valid = y_train.astype(int), y_valid.astype(int)

    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(clone(estimator), X_train, scale=scale_numeric)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        if task == "classification":
            proba = (
                pipe.predict_proba(X_valid)[:, 1]
                if hasattr(pipe.named_steps["model"], "predict_proba")
                else None
            )
            metrics = classification_metrics(y_valid, pipe.predict(X_valid), proba)
        elif task == "multiclass":
            proba = (
                pipe.predict_proba(X_valid)
                if hasattr(pipe.named_steps["model"], "predict_proba")
                else None
            )
            metrics = multiclass_metrics(y_valid, pipe.predict(X_valid), proba,
                                         labels=sorted(pd.unique(y_train)))
        else:
            metrics = regression_metrics(y_valid, pipe.predict(X_valid))
        rows.append({"model": name, **metrics})

    primary = {"classification": "pr_auc", "multiclass": "macro_f1", "regression": "mae"}[task]
    table = pd.DataFrame(rows)
    if primary in table.columns:
        table = table.sort_values(primary, ascending=(task == "regression"))
    return table.reset_index(drop=True), fitted


def baseline_classifiers(y_train: pd.Series) -> Dict[str, object]:
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "baseline_prior": DummyClassifier(strategy="stratified", random_state=cfg.RANDOM_STATE),
    }


def cluster_scan(X_scaled, k_values=range(2, 9), random_state: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Silhouette and inertia across k. Fit on TRAIN ONLY, then score holdout."""
    from sklearn.cluster import KMeans

    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X_scaled, labels)),
            "smallest_cluster_pct": float(pd.Series(labels).value_counts(normalize=True).min() * 100),
        })
    return pd.DataFrame(rows)


def baseline_regressors() -> Dict[str, object]:
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "baseline_median": DummyRegressor(strategy="median"),
    }


# --------------------------------------------------------------------------
# Decision policies
# --------------------------------------------------------------------------
def threshold_policy(y_true, y_proba, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    """Expected cost at every candidate threshold, plus the F1-optimal one.

    Two thresholds get reported because they answer different questions:
      * min-cost   -> what a business case should use
      * max-F1     -> what a leaderboard uses
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    grid = np.unique(np.round(np.quantile(y_proba, np.linspace(0.01, 0.99, 99)), 5))

    rows = []
    for t in grid:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "flagged": int(pred.sum()),
                "flagged_pct": float(pred.mean() * 100),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "f1": f1_score(y_true, pred, zero_division=0),
                "expected_cost": float(fp * cost_fp + fn * cost_fn),
            }
        )
    table = pd.DataFrame(rows)
    table["is_min_cost"] = table["expected_cost"].eq(table["expected_cost"].min())
    table["is_max_f1"] = table["f1"].eq(table["f1"].max())
    return table


def capacity_policy(y_true, y_proba, capacity_pct: float) -> Dict[str, float]:
    """Score the policy ops can actually run: intervene on the riskiest N%.

    Thresholds move with the score distribution; capacity does not. Expressing
    the policy as "we can expedite 5% of orders a day" is what makes the model
    deployable.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    k = max(1, int(round(len(y_proba) * capacity_pct / 100)))
    order = np.argsort(-y_proba)[:k]
    captured = y_true[order].sum()
    total_positives = y_true.sum()
    base_rate = y_true.mean()
    return {
        "capacity_pct": float(capacity_pct),
        "orders_flagged": int(k),
        "threshold": float(np.sort(y_proba)[::-1][k - 1]),
        "late_orders_caught": int(captured),
        "recall_at_capacity": float(captured / total_positives) if total_positives else float("nan"),
        "precision_at_capacity": float(captured / k),
        "lift_vs_random": float((captured / k) / base_rate) if base_rate else float("nan"),
    }


def calibrate_prefit(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, method: str = "isotonic"):
    """Calibrate an already-fitted pipeline on held-out data.

    scikit-learn changed how this is spelled: `cv="prefit"` up to 1.5, then
    `FrozenEstimator` from 1.6. Both are supported so the notebooks run on
    whatever version the reader has installed.
    """
    from sklearn.calibration import CalibratedClassifierCV

    X = to_sklearn_frame(X)
    y = pd.to_numeric(y, errors="coerce").astype(int)
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6

        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(pipe, method=method, cv="prefit")
    calibrated.fit(X, y)
    return calibrated


def calibration_table(y_true, y_proba, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate per decile. Big gaps mean the score is a
    ranking, not a probability, and must not be multiplied by money."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": np.asarray(y_proba, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = (
        df.groupby("bin")
        .agg(orders=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    out["gap"] = out["observed_rate"] - out["mean_predicted"]
    return out


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
def importance_table(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, top_k: int = 20,
                     n_repeats: int = 5, scoring: str | None = None) -> pd.DataFrame:
    """Permutation importance on held-out data, in original column space.

    Preferred over `feature_importances_` because it is model-agnostic, measured
    on data the model did not train on, and reported per source column rather
    than per one-hot dummy.
    """
    X = to_sklearn_frame(X)
    result = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=n_repeats,
        random_state=cfg.RANDOM_STATE,
        n_jobs=1,
        scoring=scoring,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return table.reindex(table["importance"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


def coefficient_table(pipe: Pipeline, top_k: int = 20) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None
    names = feature_names_out(pipe)
    coefs = np.ravel(model.coef_)
    if len(names) != len(coefs):
        return None
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).head(top_k).reset_index(drop=True)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
@dataclass
class ProjectPaths:
    name: str
    data_dir: Path
    model_dir: Path
    figure_dir: Path

    def parquet(self, filename: str) -> Path:
        return self.data_dir / filename

    def model(self, filename: str = "best_model.joblib") -> Path:
        return self.model_dir / filename

    def figure(self, filename: str) -> Path:
        return self.figure_dir / filename


def get_paths(project_name: str) -> ProjectPaths:
    paths = ProjectPaths(
        name=project_name,
        data_dir=cfg.ML_DIR / project_name,
        model_dir=cfg.MODELS_DIR / project_name,
        figure_dir=cfg.MODEL_FIGURES_DIR / project_name,
    )
    for directory in (paths.data_dir, paths.model_dir, paths.figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def save_artifacts(
    pipe: Pipeline,
    paths: ProjectPaths,
    target: str,
    features: Sequence[str],
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float] | None = None,
    policy: Dict[str, float] | None = None,
    notes: str = "",
    model_name: str | None = None,
) -> Dict[str, Path]:
    """Persist the model plus everything needed to reproduce and audit it."""
    model_path = paths.model("best_model.joblib")
    joblib.dump(pipe, model_path)

    payload = {
        "project": paths.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name or type(pipe.named_steps["model"]).__name__,
        "target": target,
        "n_features": len(features),
        "features": list(features),
        "metrics_test": metrics,
        "metrics_baseline_test": baseline_metrics or {},
        "decision_policy": policy or {},
        "random_state": cfg.RANDOM_STATE,
        "snapshot_date": str(cfg.SNAPSHOT_DATE.date()),
        "split": {
            "strategy": "chronological",
            "train_end": str(cfg.TRAIN_END.date()),
            "valid_end": str(cfg.VALID_END.date()),
        },
        "data_source": cfg.DATASET_SOURCE,
        "notes": notes,
    }
    meta_path = io.save_json(payload, paths.model("model_metadata.json"))
    return {"model": model_path, "metadata": meta_path}


def scorecard(project_names: Iterable[str]) -> pd.DataFrame:
    """Read every model_metadata.json back into one comparison table."""
    import json

    rows = []
    for name in project_names:
        path = cfg.MODELS_DIR / name / "model_metadata.json"
        if not path.exists():
            rows.append({"project": name, "status": "not trained"})
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "project": name,
            "status": "trained",
            "model": meta.get("model"),
            "target": meta.get("target"),
            "n_features": meta.get("n_features"),
        }
        for key, value in (meta.get("metrics_test") or {}).items():
            row[f"test_{key}"] = round(value, 4) if isinstance(value, (int, float)) else value
        for key, value in (meta.get("metrics_baseline_test") or {}).items():
            row[f"baseline_{key}"] = round(value, 4) if isinstance(value, (int, float)) else value
        rows.append(row)
    return pd.DataFrame(rows)
