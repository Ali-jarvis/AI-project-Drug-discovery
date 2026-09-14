"""Training and cross-validated evaluation.

One function, :func:`train_models`, takes a user's model selection and returns a
fitted estimator plus honest cross-validated metrics for each. Class imbalance
is handled inside the CV loop via an imbalanced-learn pipeline, so resampling
never sees the held-out fold.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .models import ModelSpec, resolve_selection

log = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    auc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float


@dataclass
class ModelResult:
    """Cross-validated performance and the final refit estimator."""

    key: str
    label: str
    family: str
    best_params: dict = field(default_factory=dict)
    fold_metrics: list[FoldMetrics] = field(default_factory=list)
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_fpr: list[float] = field(default_factory=list)
    mean_tpr: list[float] = field(default_factory=list)
    fit_seconds: float = 0.0
    estimator: object = None

    def summary(self) -> dict:
        metrics = {}
        if self.fold_metrics:
            keys = asdict(self.fold_metrics[0]).keys()
            for k in keys:
                vals = [getattr(f, k) for f in self.fold_metrics]
                metrics[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "best_params": self.best_params,
            "metrics": metrics,
            "fit_seconds": round(self.fit_seconds, 2),
        }


def _balancer(strategy: str):
    # "auto" equalises against the majority (over) or minority (under) class and
    # is a no-op on already-balanced folds; an explicit 1.0 ratio errors there.
    if strategy == "undersample":
        return ("balance", RandomUnderSampler(random_state=17, sampling_strategy="auto"))
    if strategy == "oversample":
        return ("balance", RandomOverSampler(random_state=17, sampling_strategy="auto"))
    if strategy in {"none", None}:
        return None
    raise ValueError(
        f"Unknown balance strategy {strategy!r}; use undersample, oversample or none"
    )


def build_pipeline(spec: ModelSpec, balance: str = "undersample", **model_kw) -> ImbPipeline:
    """Assemble balancing -> optional scaling -> classifier."""
    steps = []
    bal = _balancer(balance)
    if bal:
        steps.append(bal)
    if spec.needs_scaling:
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("clf", spec.build(**model_kw)))
    return ImbPipeline(steps)


def _score_fold(y_true, proba) -> FoldMetrics:
    pred = (proba >= 0.5).astype(int)
    return FoldMetrics(
        auc=float(roc_auc_score(y_true, proba)),
        average_precision=float(average_precision_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, pred)),
    )


def train_one(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    balance: str = "undersample",
    grid_search: bool = True,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> ModelResult:
    """Grid-search, cross-validate and refit a single model."""
    started = time.perf_counter()
    result = ModelResult(key=spec.key, label=spec.label, family=spec.family)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)

    pipeline = build_pipeline(spec, balance=balance)
    best_params: dict = {}

    if grid_search and spec.param_grid:
        grid = {f"clf__{k}": v for k, v in spec.param_grid.items()}
        log.info("[%s] grid search over %d parameter settings", spec.key, _grid_size(grid))
        search = GridSearchCV(
            pipeline, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=True
        )
        search.fit(X, y)
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        pipeline = search.best_estimator_
        log.info("[%s] best params: %s", spec.key, best_params)

    # Honest CV with the chosen hyperparameters: refit inside every fold.
    interp_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = build_pipeline(spec, balance=balance, **best_params)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        metrics = _score_fold(y[test_idx], proba)
        result.fold_metrics.append(metrics)

        fpr, tpr, _ = roc_curve(y[test_idx], proba)
        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        log.debug("[%s] fold %d AUC %.3f", spec.key, fold, metrics.auc)

    aucs = [m.auc for m in result.fold_metrics]
    result.mean_auc = float(np.mean(aucs))
    result.std_auc = float(np.std(aucs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    result.mean_fpr = interp_fpr.tolist()
    result.mean_tpr = mean_tpr.tolist()
    result.best_params = best_params

    # Final model sees all the data.
    final = build_pipeline(spec, balance=balance, **best_params)
    final.fit(X, y)
    result.estimator = final
    result.fit_seconds = time.perf_counter() - started

    log.info("[%s] mean AUC %.3f +/- %.3f", spec.key, result.mean_auc, result.std_auc)
    return result


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    selection: list[str] | str = "all",
    n_splits: int = 10,
    balance: str = "undersample",
    grid_search: bool = True,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> dict[str, ModelResult]:
    """Train every model the user selected."""
    specs = resolve_selection(selection)
    log.info("Training %d model(s): %s", len(specs), ", ".join(s.key for s in specs))

    results: dict[str, ModelResult] = {}
    for spec in specs:
        try:
            results[spec.key] = train_one(
                spec, X, y,
                n_splits=n_splits, balance=balance, grid_search=grid_search,
                scoring=scoring, n_jobs=n_jobs,
            )
        except Exception as exc:  # one bad model must not lose the whole run
            log.error("[%s] training failed: %s", spec.key, exc)
    if not results:
        raise RuntimeError("Every selected model failed to train; see the log above.")
    return results


def save_results(results: dict[str, ModelResult], outdir: str | Path) -> Path:
    """Persist fitted models and a JSON performance report."""
    outdir = Path(outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    report = {}
    for key, result in results.items():
        joblib.dump(result.estimator, outdir / "models" / f"{key}.joblib")
        report[key] = result.summary()
        report[key]["roc"] = {"fpr": result.mean_fpr, "tpr": result.mean_tpr}

    report_path = outdir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Saved %d model(s) and report to %s", len(results), outdir)
    return report_path


def load_models(outdir: str | Path, keys: list[str] | None = None) -> dict[str, object]:
    """Load fitted estimators previously written by :func:`save_results`."""
    models_dir = Path(outdir) / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"No models directory under {outdir}")

    paths = sorted(models_dir.glob("*.joblib"))
    if keys:
        wanted = set(keys)
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"No saved model(s) for: {', '.join(sorted(missing))}")
    if not paths:
        raise FileNotFoundError(f"No .joblib models found in {models_dir}")

    return {p.stem: joblib.load(p) for p in paths}
