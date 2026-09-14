"""Configuration.

A single YAML file drives a whole run. Every field has a default, so a minimal
config only needs to name the target and the models to train.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TargetConfig:
    uniprot_id: str | None = None
    chembl_id: str | None = None
    activity_type: str = "IC50"
    units: str = "nM"
    threshold: float = 1000.0
    ambiguous_margin: float = 0.0


@dataclass
class FeatureConfig:
    method: str = "morgan"
    n_bits: int = 1024
    radius: int = 2


@dataclass
class TrainingConfig:
    models: list[str] = field(default_factory=lambda: ["rf", "svm", "lr", "mlp"])
    cv_folds: int = 10
    balance: str = "undersample"  # undersample | oversample | none
    grid_search: bool = True
    scoring: str = "roc_auc"
    n_jobs: int = -1


@dataclass
class ScreeningConfig:
    library: str | None = None
    smiles_col: str = "smiles"
    id_col: str | None = None
    consensus: str = "mean_rank"
    weight_by_auc: bool = False
    top_fraction: float = 0.01
    top_n: int | None = None
    include_per_model_top: bool = True
    batch_size: int = 50_000


@dataclass
class Config:
    project: str = "mlvs-run"
    outdir: str = "results"
    theme: str = "light"
    target: TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        known = {"target": TargetConfig, "features": FeatureConfig,
                 "training": TrainingConfig, "screening": ScreeningConfig}
        kwargs: dict[str, Any] = {
            k: v for k, v in raw.items() if k in {"project", "outdir", "theme"}
        }
        for name, klass in known.items():
            section = raw.get(name) or {}
            unexpected = set(section) - {f for f in klass.__dataclass_fields__}
            if unexpected:
                raise ValueError(
                    f"Unknown key(s) in '{name}': {', '.join(sorted(unexpected))}"
                )
            kwargs[name] = klass(**section)
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False))
        return path

    def run_dir(self) -> Path:
        return Path(self.outdir) / self.project
