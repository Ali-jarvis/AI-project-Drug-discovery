"""Unit tests for the mlvs pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlvs import (
    Config,
    available_featurisers,
    available_models,
    consensus,
    featurise,
    resolve_selection,
    score_library,
    select_top,
    train_models,
)
from mlvs.data import clean_bioactivities, label_by_threshold, load_library
from mlvs.evaluate import metrics_table
from mlvs.screen import add_properties
from mlvs.train import load_models, save_results

ACTIVE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "COc1cc2c(cc1OC)CCN(C)C2",
    "Clc1ccccc1C2=NCC(=O)Nc3ccccc23", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "CN1CCC[C@H]1c1cccnc1", "OC(=O)c1ccccc1O",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1", "CC(N)Cc1ccccc1",
    "NC(=O)c1ccccc1", "COc1ccc2nc(S(=O)Cc3ncccc3C)[nH]c2c1",
]
INACTIVE_SMILES = [
    "CCO", "CCCC", "C1CCCCC1", "CCOCC", "CC(C)O", "CCCCO",
    "c1ccccc1", "CCCCCC", "CC(C)(C)O", "OCCO", "CCCCCCCC", "CCCCCCCCCC",
]


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(len(ACTIVE_SMILES) + len(INACTIVE_SMILES))],
            "smiles": ACTIVE_SMILES + INACTIVE_SMILES,
            "label": [1] * len(ACTIVE_SMILES) + [0] * len(INACTIVE_SMILES),
        }
    )


# -- featurisation ---------------------------------------------------------


def test_every_featuriser_produces_a_matrix(dataset):
    for method in available_featurisers():
        X, mask = featurise(dataset["smiles"].tolist(), method=method, n_bits=256)
        assert mask.all()
        assert X.shape[0] == len(dataset)
        assert X.shape[1] > 0
        assert np.isfinite(X).all()


def test_invalid_smiles_are_masked_out():
    X, mask = featurise(["CCO", "not-a-molecule", "c1ccccc1"], n_bits=64)
    assert mask.tolist() == [True, False, True]
    assert X.shape == (2, 64)


def test_unknown_featuriser_raises():
    with pytest.raises(KeyError, match="Unknown featuriser"):
        featurise(["CCO"], method="does_not_exist")


def test_morgan_radius_changes_representation():
    r1, _ = featurise(["CC(=O)Oc1ccccc1C(=O)O"], n_bits=512, radius=1)
    r3, _ = featurise(["CC(=O)Oc1ccccc1C(=O)O"], n_bits=512, radius=3)
    assert not np.array_equal(r1, r3)


# -- model registry --------------------------------------------------------


def test_registry_has_core_models():
    keys = {spec.key for spec in available_models(include_unavailable=True)}
    assert {"rf", "svm", "lr", "mlp", "xgb", "lgbm", "dnn"} <= keys


def test_resolve_selection_shortcuts():
    assert len(resolve_selection("all")) >= 7
    assert all(s.family == "classical" for s in resolve_selection("classical"))
    assert [s.key for s in resolve_selection(["rf", "lr"])] == ["rf", "lr"]


def test_resolve_selection_deduplicates():
    assert [s.key for s in resolve_selection(["rf", "rf", "lr"])] == ["rf", "lr"]


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        resolve_selection(["bogus"])


def test_empty_selection_raises():
    with pytest.raises(ValueError):
        resolve_selection([])


# -- training --------------------------------------------------------------


def test_train_selected_models(dataset):
    X, mask = featurise(dataset["smiles"].tolist(), n_bits=256)
    y = dataset.loc[mask, "label"].to_numpy()
    results = train_models(X, y, selection=["rf", "lr"], n_splits=3, grid_search=False)

    assert set(results) == {"rf", "lr"}
    for result in results.values():
        assert len(result.fold_metrics) == 3
        assert 0.0 <= result.mean_auc <= 1.0
        assert result.estimator is not None
        assert len(result.mean_fpr) == len(result.mean_tpr) == 100


def test_training_report_roundtrip(dataset, tmp_path):
    X, mask = featurise(dataset["smiles"].tolist(), n_bits=256)
    y = dataset.loc[mask, "label"].to_numpy()
    results = train_models(X, y, selection=["rf"], n_splits=3, grid_search=False)

    save_results(results, tmp_path)
    assert (tmp_path / "training_report.json").exists()

    loaded = load_models(tmp_path)
    assert "rf" in loaded
    assert loaded["rf"].predict_proba(X).shape == (len(y), 2)


def test_metrics_table_sorted_by_auc(dataset):
    X, mask = featurise(dataset["smiles"].tolist(), n_bits=256)
    y = dataset.loc[mask, "label"].to_numpy()
    results = train_models(X, y, selection=["rf", "nb"], n_splits=3, grid_search=False)

    table = metrics_table(results)
    assert list(table["auc_mean"]) == sorted(table["auc_mean"], reverse=True)
    assert {"f1_mean", "mcc_mean", "recall_mean"} <= set(table.columns)


def test_balance_strategies_all_run(dataset):
    X, mask = featurise(dataset["smiles"].tolist(), n_bits=256)
    y = dataset.loc[mask, "label"].to_numpy()
    for strategy in ("undersample", "oversample", "none"):
        results = train_models(
            X, y, selection=["lr"], n_splits=3, grid_search=False, balance=strategy
        )
        assert results["lr"].estimator is not None


# -- screening and consensus ----------------------------------------------


@pytest.fixture
def screened(dataset):
    X, mask = featurise(dataset["smiles"].tolist(), n_bits=256)
    y = dataset.loc[mask, "label"].to_numpy()
    results = train_models(X, y, selection=["rf", "lr", "nb"], n_splits=3, grid_search=False)
    models = {k: r.estimator for k, r in results.items()}
    library = pd.DataFrame(
        {
            "compound_id": [f"L{i:03d}" for i in range(len(ACTIVE_SMILES + INACTIVE_SMILES))],
            "smiles": ACTIVE_SMILES + INACTIVE_SMILES,
        }
    )
    return score_library(models, library, n_bits=256), list(models)


def test_score_library_adds_one_column_per_model(screened):
    scored, keys = screened
    for key in keys:
        assert f"{key}_score" in scored.columns
        assert scored[f"{key}_score"].between(0, 1).all()


@pytest.mark.parametrize(
    "method", ["mean_rank", "median_rank", "mean_score", "rank_product", "borda"]
)
def test_every_consensus_method_ranks_completely(screened, method):
    scored, keys = screened
    ranked = consensus(scored, keys, method=method)
    assert ranked["consensus_rank"].tolist() == list(range(1, len(ranked) + 1))
    assert ranked["compound_id"].nunique() == len(ranked)


def test_unknown_consensus_method_raises(screened):
    scored, keys = screened
    with pytest.raises(ValueError, match="Unknown consensus method"):
        consensus(scored, keys, method="telepathy")


def test_auc_weighting_changes_nothing_when_uniform(screened):
    scored, keys = screened
    flat = consensus(scored, keys, method="mean_rank")
    weighted = consensus(scored, keys, method="mean_rank", weights={k: 1.0 for k in keys})
    assert flat["compound_id"].tolist() == weighted["compound_id"].tolist()


def test_select_top_returns_unique_compounds(screened):
    scored, keys = screened
    ranked = consensus(scored, keys)
    top = select_top(ranked, n=5, model_keys=keys)
    assert top["compound_id"].is_unique
    assert 5 <= len(top) <= 5 * (len(keys) + 1)


def test_add_properties_computes_lipinski(screened):
    scored, keys = screened
    top = add_properties(consensus(scored, keys).head(5))
    assert {"MolWt", "MolLogP", "QED", "LipinskiViolations"} <= set(top.columns)
    assert top["LipinskiViolations"].between(0, 4).all()


# -- data handling ---------------------------------------------------------


def test_label_by_threshold_splits_on_potency():
    df = pd.DataFrame(
        {
            "molecule_chembl_id": list("abcd"),
            "canonical_smiles": ["CCO"] * 4,
            "standard_value": [10.0, 500.0, 5000.0, 20000.0],
        }
    )
    actives, inactives = label_by_threshold(df, threshold=1000.0)
    assert len(actives) == 2 and len(inactives) == 2
    assert set(actives["label"]) == {1}


def test_ambiguous_margin_drops_the_band():
    df = pd.DataFrame(
        {
            "molecule_chembl_id": list("abc"),
            "canonical_smiles": ["CCO"] * 3,
            "standard_value": [10.0, 1000.0, 20000.0],
        }
    )
    actives, inactives = label_by_threshold(df, threshold=1000.0, ambiguous_margin=0.5)
    assert len(actives) + len(inactives) == 2


def test_clean_collapses_duplicate_measurements():
    df = pd.DataFrame(
        {
            "molecule_chembl_id": ["a", "a", "b"],
            "canonical_smiles": ["CCO", "CCO", "CCC"],
            "standard_value": [100.0, 300.0, 50.0],
        }
    )
    cleaned = clean_bioactivities(df)
    assert len(cleaned) == 2
    assert cleaned.loc[cleaned["molecule_chembl_id"] == "a", "standard_value"].item() == 200.0


def test_load_library_normalises_columns(tmp_path):
    path = tmp_path / "lib.csv"
    pd.DataFrame({"zinc_id": ["Z1", "Z2"], "SMILES": ["CCO", "CCC"]}).to_csv(path, index=False)
    library = load_library(path, smiles_col="SMILES")
    assert list(library.columns) == ["compound_id", "smiles"]
    assert library["compound_id"].tolist() == ["Z1", "Z2"]


def test_load_library_missing_column_raises(tmp_path):
    path = tmp_path / "lib.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_library(path, smiles_col="smiles")


# -- config ----------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.training.models = ["rf", "xgb"]
    cfg.target.threshold = 500.0
    path = cfg.to_yaml(tmp_path / "c.yaml")

    reloaded = Config.from_yaml(path)
    assert reloaded.training.models == ["rf", "xgb"]
    assert reloaded.target.threshold == 500.0


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_dict({"training": {"modles": ["rf"]}})


# ---------------------------------------------------------------------------
# regression tests
# ---------------------------------------------------------------------------


def test_nested_grid_params_reach_wrapped_estimator():
    """Grid keys such as ``estimator__C`` address a wrapped sub-estimator.

    They cannot be passed to the builder as constructor arguments; the spec
    must apply them with ``set_params`` after the estimator is built.
    """
    spec = {s.key: s for s in available_models()}["linear_svm"]
    built = spec.build(estimator__C=0.1)
    assert built.get_params()["estimator__C"] == 0.1


def test_calibrated_models_survive_a_grid_search(dataset):
    """linear_svm refits from its own best params without raising."""
    results = train_models(
        *_features_and_labels(dataset),
        selection=["linear_svm"],
        n_splits=3,
    )
    assert "linear_svm" in results
    assert 0.0 <= results["linear_svm"].mean_auc <= 1.0
    assert results["linear_svm"].estimator is not None


def test_dnn_accepts_numpy_derived_flags(dataset):
    """DataLoader rejects numpy booleans, so drop_last must be a real bool."""
    pytest.importorskip("torch")
    results = train_models(
        *_features_and_labels(dataset),
        selection=["dnn"],
        n_splits=3,
    )
    assert "dnn" in results


def _features_and_labels(dataset):
    X, mask = featurise(dataset["smiles"], method="morgan", n_bits=256)
    return X, dataset.loc[mask, "label"].to_numpy()
