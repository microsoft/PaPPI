# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from rich.console import Console

from pappi.core.pipeline import Bootstrap, Dataset, Protocol, Split, audit
from pappi.core.ppi import Accuracy, MacroF1

METRICS = {"macro_f1": MacroF1, "accuracy": Accuracy}


## "pkg.module:name" -> object; with params, name is a factory called as name(**params).
## A .pkl/.joblib file is unpickled instead, which runs its code: load trusted files only.
def load_object(path: str, params: dict | None = None):
    if Path(path).suffix in (".pkl", ".pickle", ".joblib"):
        import joblib

        return joblib.load(path)
    mod, _, attr = path.partition(":")
    obj = getattr(importlib.import_module(mod), attr)
    return obj(**params) if params else obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pappi")
    a = ap.add_subparsers(dest="cmd", required=True).add_parser("audit")
    a.add_argument("data", help="csv/parquet/feather; empty label = deployment row")
    a.add_argument("--deploy", help="separate deployment table")
    a.add_argument("--config", help="YAML with the keys below, plus in_domain, correction, "
                   "<key>_params")
    a.add_argument("--label", default="label")
    a.add_argument("--features", default="x_", help="column prefix")
    a.add_argument("--protocols", nargs="*")
    a.add_argument("--fit-model", help="pkg.module:fit, fit(X, y) -> predict_proba")
    a.add_argument("--model", help="pkg.module:obj or a trusted .pkl/.joblib with predict_proba "
                   "(or f(X) -> probabilities), audited as is; labeled rows must be a test set "
                   "it never saw")
    a.add_argument("--fit-propensity", help="pkg.module:fit for the labeled-vs-deployment model")
    a.add_argument("--folds", type=int, help="k of the default fold splits (default 5); each fold "
                   "model trains on (k-1)/k of the labels")
    a.add_argument("--metric", choices=sorted(METRICS))
    a.add_argument("--inclusion", help="column of known inclusion probabilities")
    a.add_argument("--bootstrap", type=int, help="bootstrap replicates for the intervals")
    a.add_argument("--jobs", type=int, help="parallel processes for the bootstrap (same results)")
    a.add_argument("--light", action="store_true", default=None)
    a.add_argument("--out", help="report rows as CSV")
    args = ap.parse_args(argv)

    cfg = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    keys = ("protocols", "fit_model", "model", "fit_propensity", "folds", "metric", "jobs",
            "light", "inclusion", "bootstrap")
    cfg.update({k: v for k, v in vars(args).items() if k in keys and v is not None})
    for key in ("fit_model", "model", "fit_propensity"):
        if key in cfg:
            cfg[key] = load_object(cfg[key], cfg.pop(f"{key}_params", None))
    if "metric" in cfg:
        cfg["metric"] = METRICS[cfg["metric"]]()
    if "protocols" in cfg:
        cfg["protocols"] = [Protocol(**p) if isinstance(p, dict) else p
                            for p in cfg["protocols"]]
    if "jobs" in cfg:
        cfg["n_jobs"] = cfg.pop("jobs")
    if "bootstrap" in cfg:
        cfg["interval"] = Bootstrap(cfg.pop("bootstrap"))
    if "in_domain" in cfg:
        cfg["in_domain"] = [Split(e["name"], e.get("groups")) for e in cfg["in_domain"]]
        by_name = {sp.name: sp for sp in cfg["in_domain"]}
        if "correction" in cfg:
            cfg["correction_split"] = by_name[cfg.pop("correction")]

    data = Dataset.from_files(args.data, args.deploy, cfg.pop("label", args.label),
                              cfg.pop("features", args.features))
    import pappi.geo as geo

    ## Geo defaults apply only when the user chose no protocols.
    auto = geo.is_geo(data) and "protocols" not in cfg
    report = audit(data, progress=True, **({**geo.defaults(data), **cfg} if auto else cfg))
    if auto:
        report.notes.append("geo defaults")
    Console().print(report)
    if args.out:
        report.frame.to_csv(args.out, index=False)
    return 0
