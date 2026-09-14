"""Library screening and multi-model consensus ranking.

Scores a compound library with every trained model, then combines the per-model
rankings. Consensus is computed over whatever models the user trained, not a
hard-coded pair.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .featurize import featurise

log = logging.getLogger(__name__)

CONSENSUS_METHODS = ("mean_rank", "median_rank", "mean_score", "rank_product", "borda")


def score_library(
    models: dict[str, object],
    library: pd.DataFrame,
    featuriser: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
    batch_size: int = 50_000,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Predict an activity probability for every compound, for every model."""
    frames = []
    for start in range(0, len(library), batch_size):
        chunk = library.iloc[start : start + batch_size]
        features, mask = featurise(
            chunk[smiles_col].tolist(), method=featuriser, n_bits=n_bits, radius=radius
        )
        kept = chunk.loc[mask].reset_index(drop=True)
        if kept.empty:
            continue

        scored = kept.copy()
        for key, model in models.items():
            scored[f"{key}_score"] = model.predict_proba(features)[:, 1]
        frames.append(scored)
        log.info("Scored %d / %d compounds", min(start + batch_size, len(library)), len(library))

    if not frames:
        raise ValueError("No compounds in the library could be featurised.")
    return pd.concat(frames, ignore_index=True)


def add_ranks(scored: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    """Add a 1-based rank column per model (rank 1 = most likely active)."""
    out = scored.copy()
    for key in model_keys:
        out[f"{key}_rank"] = out[f"{key}_score"].rank(ascending=False, method="min").astype(int)
    return out


def consensus(
    scored: pd.DataFrame,
    model_keys: list[str] | None = None,
    method: str = "mean_rank",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine per-model predictions into one consensus ranking.

    ``mean_rank``    average of per-model ranks (the classic rank-averaging ensemble)
    ``median_rank``  median rank; less sensitive to one model disagreeing
    ``mean_score``   average of min-max normalised probabilities
    ``rank_product`` geometric mean of ranks; rewards agreement near the top
    ``borda``        mean of normalised Borda points, equivalent to mean_rank rescaled

    ``weights`` optionally weights models, e.g. by their cross-validated AUC.
    """
    if method not in CONSENSUS_METHODS:
        raise ValueError(
            f"Unknown consensus method {method!r}. Choose from: {', '.join(CONSENSUS_METHODS)}"
        )

    if model_keys is None:
        model_keys = [c[:-6] for c in scored.columns if c.endswith("_score")]
    if not model_keys:
        raise ValueError("No model score columns found to combine.")

    out = add_ranks(scored, model_keys)
    n = len(out)
    w = np.array([(weights or {}).get(k, 1.0) for k in model_keys], dtype=float)
    w = w / w.sum()

    ranks = np.column_stack([out[f"{k}_rank"].to_numpy(dtype=float) for k in model_keys])

    if method == "mean_rank":
        out["consensus_score"] = ranks @ w
        ascending = True
    elif method == "median_rank":
        out["consensus_score"] = np.median(ranks, axis=1)
        ascending = True
    elif method == "rank_product":
        out["consensus_score"] = np.exp(np.log(ranks) @ w)
        ascending = True
    elif method == "borda":
        out["consensus_score"] = ((n - ranks) / max(n - 1, 1)) @ w
        ascending = False
    else:  # mean_score
        norm = []
        for k in model_keys:
            s = out[f"{k}_score"].to_numpy(dtype=float)
            spread = s.max() - s.min()
            norm.append((s - s.min()) / spread if spread > 0 else np.zeros_like(s))
        out["consensus_score"] = np.column_stack(norm) @ w
        ascending = False

    out = out.sort_values("consensus_score", ascending=ascending).reset_index(drop=True)
    out["consensus_rank"] = np.arange(1, len(out) + 1)
    out.attrs["consensus_method"] = method
    out.attrs["models"] = model_keys
    return out


def select_top(
    ranked: pd.DataFrame,
    fraction: float = 0.01,
    n: int | None = None,
    include_per_model: bool = True,
    model_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Take the top hits by consensus, optionally unioned with each model's own top slice.

    Including each model's own top slice keeps candidates that one model rates
    very highly but the consensus dilutes — useful when models disagree.
    """
    cut = n if n is not None else max(1, int(len(ranked) * fraction))
    picks = [ranked.head(cut)]

    if include_per_model:
        keys = model_keys or ranked.attrs.get("models") or [
            c[:-5] for c in ranked.columns if c.endswith("_rank") and c != "consensus_rank"
        ]
        for key in keys:
            col = f"{key}_rank"
            if col in ranked.columns:
                picks.append(ranked.nsmallest(cut, col))

    top = (
        pd.concat(picks)
        .drop_duplicates(subset="compound_id")
        .sort_values("consensus_rank")
        .reset_index(drop=True)
    )
    log.info("Selected %d unique compounds (top %d per ranking)", len(top), cut)
    return top


def add_properties(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Append physicochemical properties and Lipinski / QED flags to hits."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED

    rows = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({})
            continue
        props = {
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "TPSA": Descriptors.TPSA(mol),
            "QED": QED.qed(mol),
        }
        props["LipinskiViolations"] = sum(
            [
                props["MolWt"] > 500,
                props["MolLogP"] > 5,
                props["NumHAcceptors"] > 10,
                props["NumHDonors"] > 5,
            ]
        )
        rows.append(props)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def write_results(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Wrote %d rows to %s", len(df), path)
    return path
