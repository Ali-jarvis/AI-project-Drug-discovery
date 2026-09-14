"""Dataset assembly.

Retrieves bioactivity data for a target from ChEMBL and splits it into active
and inactive sets at a user-defined potency threshold. The threshold is a
config value rather than a constant, because it is the single choice that most
changes what a screening model learns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

#: Activity types handled, and whether a *larger* value means more potent.
ACTIVITY_TYPES = {
    "IC50": False,
    "Ki": False,
    "Kd": False,
    "EC50": False,
    "Potency": False,
    "Inhibition": True,
}


@dataclass
class DatasetSummary:
    target: str
    chembl_id: str | None
    n_raw: int
    n_actives: int
    n_inactives: int
    threshold: float
    activity_type: str
    units: str

    def __str__(self) -> str:
        return (
            f"{self.target} ({self.chembl_id}): {self.n_actives} actives / "
            f"{self.n_inactives} inactives from {self.n_raw} records "
            f"at {self.activity_type} {'>' if ACTIVITY_TYPES.get(self.activity_type) else '<'} "
            f"{self.threshold} {self.units}"
        )


def uniprot_to_chembl(uniprot_id: str) -> str:
    """Resolve a UniProt accession to a single-protein ChEMBL target id."""
    from chembl_webresource_client.new_client import new_client

    hits = pd.DataFrame(
        new_client.target.filter(target_components__accession=uniprot_id)
    )
    if hits.empty:
        raise LookupError(f"No ChEMBL target found for UniProt {uniprot_id!r}")

    single = hits[hits["target_type"] == "SINGLE PROTEIN"]
    chosen = (single if not single.empty else hits).iloc[0]
    log.info(
        "UniProt %s -> %s (%s)", uniprot_id, chosen["target_chembl_id"], chosen["pref_name"]
    )
    return chosen["target_chembl_id"]


def fetch_bioactivities(
    chembl_id: str,
    activity_type: str = "IC50",
    units: str = "nM",
) -> pd.DataFrame:
    """Download activity records for a ChEMBL target."""
    from chembl_webresource_client.new_client import new_client

    log.info("Fetching %s data for %s from ChEMBL", activity_type, chembl_id)
    records = new_client.activity.filter(
        target_chembl_id=chembl_id,
        standard_type=activity_type,
        standard_units=units,
    ).only(
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "standard_type",
            "assay_chembl_id",
            "assay_description",
        ]
    )
    df = pd.DataFrame(list(records))
    if df.empty:
        raise LookupError(
            f"ChEMBL returned no {activity_type} records in {units} for {chembl_id}"
        )
    log.info("Retrieved %d raw activity records", len(df))
    return df


def clean_bioactivities(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and collapse duplicate measurements to the median."""
    before = len(df)
    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]

    # One compound can appear in many assays; take the median so a single
    # outlying measurement cannot flip a label.
    df = (
        df.groupby("molecule_chembl_id", as_index=False)
        .agg(
            canonical_smiles=("canonical_smiles", "first"),
            standard_value=("standard_value", "median"),
            n_measurements=("standard_value", "size"),
        )
        .reset_index(drop=True)
    )
    log.info("Cleaned %d records down to %d unique compounds", before, len(df))
    return df


def label_by_threshold(
    df: pd.DataFrame,
    threshold: float = 1000.0,
    activity_type: str = "IC50",
    ambiguous_margin: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split compounds into actives and inactives at ``threshold``.

    ``ambiguous_margin`` optionally discards a band around the threshold, e.g.
    0.5 drops everything between 0.5x and 2x the cutoff, where the active/inactive
    call is least reliable.
    """
    higher_is_active = ACTIVITY_TYPES.get(activity_type, False)
    values = df["standard_value"]

    if ambiguous_margin > 0:
        low, high = threshold * (1 - ambiguous_margin), threshold * (1 + ambiguous_margin)
        keep = (values <= low) | (values >= high)
        dropped = int((~keep).sum())
        if dropped:
            log.info("Dropped %d compounds inside the ambiguous band", dropped)
        df = df[keep]
        values = df["standard_value"]

    active_mask = values >= threshold if higher_is_active else values <= threshold
    actives = df[active_mask].copy()
    inactives = df[~active_mask].copy()
    actives["label"] = 1
    inactives["label"] = 0
    return actives, inactives


def build_dataset(
    uniprot_id: str | None = None,
    chembl_id: str | None = None,
    activity_type: str = "IC50",
    units: str = "nM",
    threshold: float = 1000.0,
    ambiguous_margin: float = 0.0,
) -> tuple[pd.DataFrame, DatasetSummary]:
    """Fetch, clean and label a training set for one target."""
    if not (uniprot_id or chembl_id):
        raise ValueError("Provide either uniprot_id or chembl_id")
    if chembl_id is None:
        chembl_id = uniprot_to_chembl(uniprot_id)

    raw = fetch_bioactivities(chembl_id, activity_type=activity_type, units=units)
    cleaned = clean_bioactivities(raw)
    actives, inactives = label_by_threshold(
        cleaned, threshold=threshold, activity_type=activity_type,
        ambiguous_margin=ambiguous_margin,
    )

    dataset = (
        pd.concat([actives, inactives], ignore_index=True)
        .rename(columns={"molecule_chembl_id": "compound_id", "canonical_smiles": "smiles"})
        .loc[:, ["compound_id", "smiles", "standard_value", "n_measurements", "label"]]
    )
    summary = DatasetSummary(
        target=uniprot_id or chembl_id,
        chembl_id=chembl_id,
        n_raw=len(raw),
        n_actives=len(actives),
        n_inactives=len(inactives),
        threshold=threshold,
        activity_type=activity_type,
        units=units,
    )
    log.info("%s", summary)
    return dataset, summary


def load_library(
    path: str, smiles_col: str = "smiles", id_col: str | None = None
) -> pd.DataFrame:
    """Read a screening library CSV and normalise its column names."""
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise KeyError(
            f"Column {smiles_col!r} not in {path}. Columns: {', '.join(map(str, df.columns))}"
        )
    if id_col is None:
        candidates = [c for c in df.columns if "id" in str(c).lower()]
        id_col = candidates[0] if candidates else None

    out = pd.DataFrame({"smiles": df[smiles_col].astype(str)})
    out.insert(
        0,
        "compound_id",
        df[id_col].astype(str) if id_col else [f"cmpd_{i}" for i in range(len(df))],
    )
    out = out.drop_duplicates(subset="compound_id").reset_index(drop=True)
    log.info("Loaded %d compounds from %s", len(out), path)
    return out
