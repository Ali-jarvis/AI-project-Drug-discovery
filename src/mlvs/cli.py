"""Command-line interface.

Subcommands:

    mlvs models                  list selectable models and featurisers
    mlvs init                    write a starter config file
    mlvs fetch                   build a training set from ChEMBL
    mlvs train                   train the selected models with cross-validation
    mlvs screen                  score a library and rank by consensus
    mlvs run                     fetch -> train -> screen in one go
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import Config
from .featurize import FEATURISERS, available_featurisers, featurise
from .models import available_models

log = logging.getLogger("mlvs")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(args) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    # Command-line flags win over the file.
    if getattr(args, "models", None):
        cfg.training.models = [m.strip() for m in args.models.split(",")]
    for attr, target in (
        ("uniprot", ("target", "uniprot_id")),
        ("chembl", ("target", "chembl_id")),
        ("threshold", ("target", "threshold")),
        ("featuriser", ("features", "method")),
        ("n_bits", ("features", "n_bits")),
        ("cv_folds", ("training", "cv_folds")),
        ("library", ("screening", "library")),
        ("consensus", ("screening", "consensus")),
        ("weight_by_auc", ("screening", "weight_by_auc")),
        ("top_fraction", ("screening", "top_fraction")),
        ("balance", ("training", "balance")),
        ("outdir", (None, "outdir")),
        ("project", (None, "project")),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            section, field_name = target
            setattr(getattr(cfg, section) if section else cfg, field_name, value)
    return cfg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_models(args) -> int:
    print("\nModels — select any combination by key\n")
    print(f"  {'key':<12}{'family':<11}{'ready':<8}model")
    print("  " + "-" * 76)
    for spec in available_models(include_unavailable=True):
        mark = "yes" if spec.available else "no"
        print(f"  {spec.key:<12}{spec.family:<11}{mark:<8}{spec.label}")
        note = spec.description
        if not spec.available:
            note = f"{note}  [needs: pip install {spec.requires}]"
        print(f"  {'':<31}{note}")
    print("\n  Shortcuts: all, classical, ensemble, deep\n")

    print("Featurisers\n")
    print(f"  {'key':<22}description")
    print("  " + "-" * 76)
    for name in available_featurisers():
        print(f"  {name:<22}{FEATURISERS[name].description}")
    print()
    return 0


def cmd_init(args) -> int:
    path = Path(args.output)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    Config().to_yaml(path)
    print(f"Wrote starter config to {path}")
    return 0


def cmd_fetch(args) -> int:
    from .data import build_dataset

    cfg = _load_config(args)
    dataset, summary = build_dataset(
        uniprot_id=cfg.target.uniprot_id,
        chembl_id=cfg.target.chembl_id,
        activity_type=cfg.target.activity_type,
        units=cfg.target.units,
        threshold=cfg.target.threshold,
        ambiguous_margin=cfg.target.ambiguous_margin,
    )
    out = cfg.run_dir() / "dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    print(f"\n{summary}\nWrote {out}")
    return 0


def cmd_train(args) -> int:
    from .evaluate import metrics_table, write_report
    from .train import save_results, train_models

    cfg = _load_config(args)
    run_dir = cfg.run_dir()
    dataset_path = Path(args.dataset) if args.dataset else run_dir / "dataset.csv"
    if not dataset_path.exists():
        print(f"No dataset at {dataset_path}. Run `mlvs fetch` first.", file=sys.stderr)
        return 1

    dataset = pd.read_csv(dataset_path)
    log.info("Loaded %d compounds from %s", len(dataset), dataset_path)

    X, mask = featurise(
        dataset["smiles"].tolist(),
        method=cfg.features.method,
        n_bits=cfg.features.n_bits,
        radius=cfg.features.radius,
    )
    y = dataset.loc[mask, "label"].to_numpy()
    log.info("Feature matrix %s | %d active, %d inactive", X.shape, int(y.sum()), int((y == 0).sum()))

    results = train_models(
        X, y,
        selection=cfg.training.models,
        n_splits=cfg.training.cv_folds,
        balance=cfg.training.balance,
        grid_search=cfg.training.grid_search,
        scoring=cfg.training.scoring,
        n_jobs=cfg.training.n_jobs,
    )
    save_results(results, run_dir)
    write_report(
        results, run_dir / "figures",
        theme=cfg.theme, table_path=run_dir / "model_metrics.csv",
    )
    cfg.to_yaml(run_dir / "config_used.yaml")

    print("\nCross-validated performance\n")
    table = metrics_table(results)
    print(table[["label", "auc_mean", "auc_std", "f1_mean", "mcc_mean"]].to_string(index=False))
    print(f"\nModels, figures and metrics written to {run_dir}")
    return 0


def cmd_screen(args) -> int:
    import json

    from .data import load_library
    from .screen import add_properties, consensus, score_library, select_top, write_results
    from .train import load_models

    cfg = _load_config(args)
    run_dir = cfg.run_dir()
    if not cfg.screening.library:
        print("No screening library set (--library or screening.library).", file=sys.stderr)
        return 1

    models = load_models(run_dir, keys=args.models.split(",") if args.models else None)
    log.info("Loaded models: %s", ", ".join(models))

    library = load_library(
        cfg.screening.library,
        smiles_col=cfg.screening.smiles_col,
        id_col=cfg.screening.id_col,
    )
    scored = score_library(
        models, library,
        featuriser=cfg.features.method,
        n_bits=cfg.features.n_bits,
        radius=cfg.features.radius,
        batch_size=cfg.screening.batch_size,
    )

    weights = None
    if cfg.screening.weight_by_auc:
        report_path = run_dir / "training_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            weights = {
                k: report[k]["metrics"]["auc"]["mean"] for k in models if k in report
            }
            log.info("Weighting consensus by AUC: %s", weights)

    ranked = consensus(scored, list(models), method=cfg.screening.consensus, weights=weights)
    top = select_top(
        ranked,
        fraction=cfg.screening.top_fraction,
        n=cfg.screening.top_n,
        include_per_model=cfg.screening.include_per_model_top,
        model_keys=list(models),
    )
    top = add_properties(top)

    write_results(ranked, run_dir / "screening_all.csv")
    write_results(top, run_dir / "screening_top_hits.csv")

    print(f"\nTop {len(top)} hits by {cfg.screening.consensus} consensus\n")
    cols = ["compound_id", "consensus_rank"] + [f"{k}_score" for k in models]
    print(top[cols].head(15).to_string(index=False))
    print(f"\nFull results written to {run_dir}")
    return 0


def cmd_run(args) -> int:
    for step in (cmd_fetch, cmd_train, cmd_screen):
        code = step(args)
        if code != 0:
            return code
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlvs",
        description="MLvs: machine learning based virtual screening with a selectable model set.",
    )
    parser.add_argument("--version", action="version", version=f"mlvs {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p, *, needs_config=True):
        if needs_config:
            p.add_argument("-c", "--config", help="path to a YAML config file")
            p.add_argument("--project", help="run name (overrides config)")
            p.add_argument("--outdir", help="output directory (overrides config)")
        return p

    p = sub.add_parser("models", help="list selectable models and featurisers")
    p.set_defaults(func=cmd_models)

    p = shared(sub.add_parser("init", help="write a starter config"), needs_config=False)
    p.add_argument("-o", "--output", default="config.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = shared(sub.add_parser("fetch", help="build a training set from ChEMBL"))
    p.add_argument("--uniprot", help="UniProt accession, e.g. P27487")
    p.add_argument("--chembl", help="ChEMBL target id, e.g. CHEMBL284")
    p.add_argument("--threshold", type=float, help="activity cutoff (default 1000 nM)")
    p.set_defaults(func=cmd_fetch)

    p = shared(sub.add_parser("train", help="train the selected models"))
    p.add_argument("--dataset", help="training CSV (defaults to the run's dataset.csv)")
    p.add_argument("--models", help="comma-separated keys, or all/classical/ensemble/deep")
    p.add_argument("--featuriser", help="molecular representation")
    p.add_argument("--n-bits", type=int, dest="n_bits")
    p.add_argument("--cv-folds", type=int, dest="cv_folds")
    p.add_argument("--balance", help="undersample | oversample | none")
    p.set_defaults(func=cmd_train)

    p = shared(sub.add_parser("screen", help="score a library and rank by consensus"))
    p.add_argument("--library", help="library CSV to screen")
    p.add_argument("--models", help="restrict to these trained models")
    p.add_argument("--consensus", help="mean_rank | median_rank | mean_score | rank_product | borda")
    p.add_argument("--weight-by-auc", dest="weight_by_auc", action="store_true", default=None,
                   help="weight each model's contribution by its cross-validated AUC")
    p.add_argument("--top-fraction", type=float, dest="top_fraction",
                   help="fraction of the library to keep as hits, e.g. 0.01")
    p.set_defaults(func=cmd_screen)

    p = shared(sub.add_parser("run", help="fetch, train and screen in one command"))
    p.add_argument("--uniprot")
    p.add_argument("--chembl")
    p.add_argument("--threshold", type=float)
    p.add_argument("--models")
    p.add_argument("--featuriser")
    p.add_argument("--n-bits", type=int, dest="n_bits")
    p.add_argument("--cv-folds", type=int, dest="cv_folds")
    p.add_argument("--library")
    p.add_argument("--consensus")
    p.add_argument("--weight-by-auc", dest="weight_by_auc", action="store_true", default=None,
                   help="weight each model's contribution by its cross-validated AUC")
    p.add_argument("--top-fraction", type=float, dest="top_fraction",
                   help="fraction of the library to keep as hits, e.g. 0.01")
    p.add_argument("--balance", help="undersample | oversample | none")
    p.add_argument("--dataset", default=None)
    p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        log.error("%s", exc)
        if args.verbose:
            raise
        print("\nRe-run with -v for the full traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
