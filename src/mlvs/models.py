"""Model registry.

Every classifier the pipeline can train is declared here as a :class:`ModelSpec`.
Users select models by name in the config file or on the command line; nothing
else in the codebase needs to know which algorithms exist.

To add a model, append one :class:`ModelSpec` to ``_SPECS``. Optional
dependencies (XGBoost, LightGBM, CatBoost, PyTorch) are declared with a
``requires`` string and are only imported when actually selected, so the package
works with scikit-learn alone.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

RANDOM_STATE = 17


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of a selectable classifier."""

    key: str
    label: str
    family: str  # "classical" | "ensemble" | "deep"
    builder: Callable[..., Any]
    param_grid: dict[str, list] = field(default_factory=dict)
    requires: str | None = None  # optional pip package
    needs_scaling: bool = False
    description: str = ""

    @property
    def available(self) -> bool:
        if self.requires is None:
            return True
        return importlib.util.find_spec(self.requires) is not None

    def build(self, **overrides) -> Any:
        if not self.available:
            raise ImportError(
                f"Model {self.key!r} needs the optional dependency {self.requires!r}. "
                f"Install it with: pip install {self.requires}"
            )
        # Grid-search results may carry nested keys such as ``estimator__C``
        # that address a wrapped sub-estimator. Those cannot go to the builder
        # as constructor arguments; apply them with ``set_params`` instead.
        nested = {k: v for k, v in overrides.items() if "__" in k}
        direct = {k: v for k, v in overrides.items() if "__" not in k}
        estimator = self.builder(**direct)
        if nested:
            estimator.set_params(**nested)
        return estimator


# --------------------------------------------------------------------------
# scikit-learn builders
# --------------------------------------------------------------------------


def _random_forest(**kw):
    from sklearn.ensemble import RandomForestClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_jobs", -1)
    return RandomForestClassifier(**kw)


def _extra_trees(**kw):
    from sklearn.ensemble import ExtraTreesClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_jobs", -1)
    return ExtraTreesClassifier(**kw)


def _gradient_boosting(**kw):
    from sklearn.ensemble import GradientBoostingClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    return GradientBoostingClassifier(**kw)


def _svm(**kw):
    from sklearn.svm import SVC

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("probability", True)
    return SVC(**kw)


def _linear_svm(**kw):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("max_iter", 5000)
    # LinearSVC has no predict_proba; calibrate so ranking stays comparable.
    return CalibratedClassifierCV(LinearSVC(**kw), cv=3)


def _logistic_regression(**kw):
    from sklearn.linear_model import LogisticRegression

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("max_iter", 2000)
    return LogisticRegression(**kw)


def _mlp(**kw):
    from sklearn.neural_network import MLPClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("max_iter", 500)
    return MLPClassifier(**kw)


def _knn(**kw):
    from sklearn.neighbors import KNeighborsClassifier

    kw.setdefault("n_jobs", -1)
    return KNeighborsClassifier(**kw)


def _naive_bayes(**kw):
    from sklearn.naive_bayes import BernoulliNB

    return BernoulliNB(**kw)


# --------------------------------------------------------------------------
# Optional gradient-boosting libraries
# --------------------------------------------------------------------------


def _xgboost(**kw):
    from xgboost import XGBClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("eval_metric", "logloss")
    kw.setdefault("n_jobs", -1)
    return XGBClassifier(**kw)


def _lightgbm(**kw):
    from lightgbm import LGBMClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("n_jobs", -1)
    kw.setdefault("verbose", -1)
    return LGBMClassifier(**kw)


def _catboost(**kw):
    from catboost import CatBoostClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    kw.setdefault("verbose", False)
    return CatBoostClassifier(**kw)


# --------------------------------------------------------------------------
# Deep learning
# --------------------------------------------------------------------------


def _torch_dnn(**kw):
    from .deep import TorchDNNClassifier

    kw.setdefault("random_state", RANDOM_STATE)
    return TorchDNNClassifier(**kw)


_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="rf",
        label="Random forest",
        family="ensemble",
        builder=_random_forest,
        param_grid={"n_estimators": [50, 100, 200], "max_depth": [4, 6, 10, 12]},
        description="Bagged decision trees; robust default for fingerprint data.",
    ),
    ModelSpec(
        key="et",
        label="Extra trees",
        family="ensemble",
        builder=_extra_trees,
        param_grid={"n_estimators": [100, 200], "max_depth": [6, 12, None]},
        description="Extremely randomised trees; faster to fit than RF.",
    ),
    ModelSpec(
        key="gbm",
        label="Gradient boosting",
        family="ensemble",
        builder=_gradient_boosting,
        param_grid={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        description="scikit-learn gradient boosted trees.",
    ),
    ModelSpec(
        key="svm",
        label="Support vector machine (RBF)",
        family="classical",
        builder=_svm,
        param_grid={"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
        needs_scaling=True,
        description="Kernel SVM with probability calibration; strong on 1024-bit Morgan FPs.",
    ),
    ModelSpec(
        key="linear_svm",
        label="Linear SVM (calibrated)",
        family="classical",
        builder=_linear_svm,
        param_grid={"estimator__C": [0.01, 0.1, 1]},
        needs_scaling=True,
        description="Linear SVM wrapped in probability calibration; scales to large sets.",
    ),
    ModelSpec(
        key="lr",
        label="Logistic regression",
        family="classical",
        builder=_logistic_regression,
        param_grid={"C": [0.01, 0.1, 1, 10]},
        needs_scaling=True,
        description="Regularised linear baseline.",
    ),
    ModelSpec(
        key="mlp",
        label="Multilayer perceptron",
        family="deep",
        builder=_mlp,
        param_grid={
            "hidden_layer_sizes": [(50, 50, 50), (50, 50), (50,)],
            "activation": ["tanh", "relu"],
            "alpha": [0.0001, 0.01],
        },
        needs_scaling=True,
        description="scikit-learn feed-forward neural network.",
    ),
    ModelSpec(
        key="knn",
        label="k-nearest neighbours",
        family="classical",
        builder=_knn,
        param_grid={"n_neighbors": [3, 5, 11], "metric": ["jaccard", "euclidean"]},
        description="Similarity baseline; Jaccard metric suits binary fingerprints.",
    ),
    ModelSpec(
        key="nb",
        label="Bernoulli naive Bayes",
        family="classical",
        builder=_naive_bayes,
        param_grid={"alpha": [0.1, 1.0]},
        description="Fast probabilistic baseline for binary features.",
    ),
    ModelSpec(
        key="xgb",
        label="XGBoost",
        family="ensemble",
        builder=_xgboost,
        param_grid={"n_estimators": [200, 400], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
        requires="xgboost",
        description="Gradient boosted trees; often the strongest classical option.",
    ),
    ModelSpec(
        key="lgbm",
        label="LightGBM",
        family="ensemble",
        builder=_lightgbm,
        param_grid={"n_estimators": [200, 400], "num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
        requires="lightgbm",
        description="Histogram-based boosting; fast on large libraries.",
    ),
    ModelSpec(
        key="catboost",
        label="CatBoost",
        family="ensemble",
        builder=_catboost,
        param_grid={"iterations": [300, 600], "depth": [4, 6]},
        requires="catboost",
        description="Ordered boosting with strong defaults.",
    ),
    ModelSpec(
        key="dnn",
        label="Deep neural network (PyTorch)",
        family="deep",
        builder=_torch_dnn,
        param_grid={
            "hidden_sizes": [(512, 256), (1024, 512, 256)],
            "dropout": [0.2, 0.4],
            "lr": [1e-3],
        },
        requires="torch",
        needs_scaling=True,
        description="Configurable feed-forward net with dropout and early stopping.",
    ),
]

REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}


def get_model(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown model {key!r}. Available: {', '.join(sorted(REGISTRY))}. "
            "Run `mlvs models` to see descriptions and availability."
        )
    return REGISTRY[key]


def available_models(include_unavailable: bool = False) -> list[ModelSpec]:
    specs = sorted(REGISTRY.values(), key=lambda s: (s.family, s.key))
    if include_unavailable:
        return specs
    return [s for s in specs if s.available]


def resolve_selection(selection: list[str] | str) -> list[ModelSpec]:
    """Turn a user selection into model specs.

    Accepts a list of keys, the string ``"all"`` for everything installed, or a
    family name (``classical``, ``ensemble``, ``deep``).
    """
    if isinstance(selection, str):
        selection = [selection]

    resolved: list[ModelSpec] = []
    for item in selection:
        item = item.strip().lower()
        if item == "all":
            resolved.extend(available_models())
        elif item in {"classical", "ensemble", "deep"}:
            resolved.extend(s for s in available_models() if s.family == item)
        else:
            spec = get_model(item)
            if not spec.available:
                raise ImportError(
                    f"Model {item!r} requires the optional package {spec.requires!r}. "
                    f"Install it with: pip install {spec.requires}"
                )
            resolved.append(spec)

    seen, unique = set(), []
    for spec in resolved:
        if spec.key not in seen:
            seen.add(spec.key)
            unique.append(spec)
    if not unique:
        raise ValueError("Model selection resolved to an empty set.")
    return unique
