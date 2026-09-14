"""MLvs — machine learning based virtual screening.

Written for the DPP-4 natural product screening described in
Ali et al., Int. J. Mol. Sci. 2026, 27, 5536 (doi:10.3390/ijms27125536),
and generalised so any target, featuriser and model combination can be run.

Typical use::

    from mlvs import featurise, train_models, score_library, consensus

    X, mask = featurise(df["smiles"], method="morgan", n_bits=1024)
    results = train_models(X, df.loc[mask, "label"], selection=["rf", "svm", "mlp"])
    scored = score_library({k: r.estimator for k, r in results.items()}, library)
    hits = consensus(scored, method="mean_rank")
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Shahid Ali"

from .config import Config
from .featurize import FEATURISERS, available_featurisers, featurise, featurise_frame
from .models import REGISTRY, ModelSpec, available_models, get_model, resolve_selection
from .screen import add_properties, consensus, score_library, select_top
from .train import ModelResult, load_models, save_results, train_models, train_one

__all__ = [
    "__version__",
    "Config",
    "FEATURISERS",
    "REGISTRY",
    "ModelResult",
    "ModelSpec",
    "add_properties",
    "available_featurisers",
    "available_models",
    "consensus",
    "featurise",
    "featurise_frame",
    "get_model",
    "load_models",
    "resolve_selection",
    "save_results",
    "score_library",
    "select_top",
    "train_models",
    "train_one",
]
