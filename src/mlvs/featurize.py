"""Molecular featurisation.

Turns SMILES strings into numeric feature matrices. Every featuriser is
registered in ``FEATURISERS`` and addressed by name from the config file, so
adding a new representation means adding one function and one registry entry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator,
)

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class Featuriser:
    """A named molecular representation."""

    name: str
    fn: Callable[[Sequence[Chem.Mol], int, int], np.ndarray]
    supports_n_bits: bool = True
    supports_radius: bool = False
    description: str = ""


def _fp_to_array(generator, mols: Sequence[Chem.Mol]) -> np.ndarray:
    out = np.zeros((len(mols), generator.GetOptions().fpSize), dtype=np.uint8)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        out[i] = np.frombuffer(
            generator.GetFingerprintAsNumPy(mol).tobytes(), dtype=np.uint8
        )
    return out


def morgan(mols, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(
        [
            gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(n_bits)
            for m in mols
        ],
        dtype=np.uint8,
    )


def atom_pair(mols, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    gen = GetAtomPairGenerator(fpSize=n_bits)
    return np.array(
        [
            gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(n_bits)
            for m in mols
        ],
        dtype=np.uint8,
    )


def topological_torsion(mols, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    gen = GetTopologicalTorsionGenerator(fpSize=n_bits)
    return np.array(
        [
            gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(n_bits)
            for m in mols
        ],
        dtype=np.uint8,
    )


def rdkit_fp(mols, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    gen = GetRDKitFPGenerator(fpSize=n_bits)
    return np.array(
        [
            gen.GetFingerprintAsNumPy(m) if m is not None else np.zeros(n_bits)
            for m in mols
        ],
        dtype=np.uint8,
    )


def maccs(mols, n_bits: int = 167, radius: int = 2) -> np.ndarray:
    return np.array(
        [
            np.array(MACCSkeys.GenMACCSKeys(m)) if m is not None else np.zeros(167)
            for m in mols
        ],
        dtype=np.uint8,
    )


#: Physicochemical descriptors used by the ``descriptors`` featuriser.
DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "TPSA",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumAromaticRings",
]


def descriptors(mols, n_bits: int = 0, radius: int = 2) -> np.ndarray:
    fns = {name: getattr(Descriptors, name) for name in DESCRIPTOR_NAMES}
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([0.0] * len(DESCRIPTOR_NAMES))
            continue
        rows.append([float(fn(mol)) for fn in fns.values()])
    return np.asarray(rows, dtype=np.float32)


def morgan_plus_descriptors(mols, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    return np.hstack([morgan(mols, n_bits, radius), descriptors(mols)]).astype(
        np.float32
    )


FEATURISERS: dict[str, Featuriser] = {
    "morgan": Featuriser(
        "morgan", morgan, supports_radius=True, description="Morgan / ECFP circular fingerprint"
    ),
    "atom_pair": Featuriser("atom_pair", atom_pair, description="Atom-pair fingerprint"),
    "topological_torsion": Featuriser(
        "topological_torsion", topological_torsion, description="Topological torsion fingerprint"
    ),
    "rdkit": Featuriser("rdkit", rdkit_fp, description="RDKit path-based fingerprint"),
    "maccs": Featuriser(
        "maccs", maccs, supports_n_bits=False, description="MACCS 166 structural keys"
    ),
    "descriptors": Featuriser(
        "descriptors",
        descriptors,
        supports_n_bits=False,
        description=f"{len(DESCRIPTOR_NAMES)} physicochemical descriptors",
    ),
    "morgan+descriptors": Featuriser(
        "morgan+descriptors",
        morgan_plus_descriptors,
        supports_radius=True,
        description="Morgan fingerprint concatenated with physicochemical descriptors",
    ),
}


def available_featurisers() -> list[str]:
    return sorted(FEATURISERS)


def parse_smiles(smiles: Iterable[str]) -> tuple[list[Chem.Mol], np.ndarray]:
    """Parse SMILES, returning the molecules and a boolean mask of valid entries."""
    mols, valid = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        mols.append(mol)
        valid.append(mol is not None)
    mask = np.asarray(valid, dtype=bool)
    n_bad = int((~mask).sum())
    if n_bad:
        log.warning("%d SMILES could not be parsed and will be dropped", n_bad)
    return mols, mask


def featurise(
    smiles: Sequence[str],
    method: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
    drop_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Featurise SMILES.

    Returns the feature matrix and the boolean mask of rows that parsed. When
    ``drop_invalid`` is set the matrix contains only valid rows, so callers must
    apply the mask to any parallel arrays (labels, identifiers).
    """
    if method not in FEATURISERS:
        raise KeyError(
            f"Unknown featuriser {method!r}. Available: {', '.join(available_featurisers())}"
        )
    spec = FEATURISERS[method]
    mols, mask = parse_smiles(smiles)
    features = spec.fn(mols, n_bits, radius)
    if drop_invalid:
        features = features[mask]
    return features, mask


def featurise_frame(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    method: str = "morgan",
    n_bits: int = 1024,
    radius: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Featurise a dataframe, returning the surviving rows and their features."""
    features, mask = featurise(
        df[smiles_col].tolist(), method=method, n_bits=n_bits, radius=radius
    )
    return df.loc[mask].reset_index(drop=True), features
