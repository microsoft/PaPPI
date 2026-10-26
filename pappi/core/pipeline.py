# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from typing import Protocol as Contract

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from pappi.core.ppi import (
    _PI_CLIP_MIN,
    MacroF1,
    Metric,
    inclusion_weights,
    inverse_odds_weights,
    ppi_estimate,
)

N_FOLDS = 5
LIGHT_U, LIGHT_ROUNDS = 2_000, 50
ARMS = {"x": "PPI, propensity on X", "z": "PPI, propensity on Z",
        "xz": "PPI, propensity on [X, Z]", "pi": "PPI, known inclusion"}


## Contracts the engine relies on (structural: any object with these methods fits).
class Fitted(Contract):
    def predict_proba(self, X) -> np.ndarray: ...


## fit(X, y, seed) -> Fitted; predict_proba columns follow the classes 0..K-1 of y.
FitFn = Callable[[np.ndarray, np.ndarray, int], Fitted]


## A Z covariate or a group spec: a column name, or (name, fn) with fn(frame) -> array.
Spec = str | tuple[str, Callable[[pd.DataFrame], np.ndarray]]
## assign(n_rows, groups, seed) -> fold id per row; rows sharing a group share a fold.
FoldFn = Callable[[int, np.ndarray | None, int], np.ndarray]


def read_table(path, **kw) -> pd.DataFrame:
    readers = {".csv": pd.read_csv, ".parquet": pd.read_parquet, ".feather": pd.read_feather}
    return readers[Path(path).suffix](path, **kw)


## A row with an empty label is a deployment sample (U); the rest are labeled (L).
@dataclass
class Dataset:
    frame: pd.DataFrame
    features: list[str]
    label: str = "label"

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, label: str = "label",
                   features: str | list[str] = "x_") -> Dataset:
        cols = ([c for c in frame.columns if c.startswith(features)]
                if isinstance(features, str) else list(features))
        if not cols or label not in frame.columns:
            raise ValueError(f"need feature columns {features!r} and a {label!r} column")
        return cls(frame.reset_index(drop=True), cols, label)

    @classmethod
    def from_frames(cls, labeled: pd.DataFrame, deploy: pd.DataFrame, label: str = "label",
                    features: str | list[str] = "x_") -> Dataset:
        frame = pd.concat([labeled, deploy.assign(**{label: np.nan})], ignore_index=True)
        return cls.from_frame(frame, label, features)

    @classmethod
    def from_files(cls, labeled, deploy=None, label: str = "label",
                   features: str | list[str] = "x_") -> Dataset:
        if deploy is not None:
            return cls.from_frames(read_table(labeled), read_table(deploy), label, features)
        kw = {"dtype": {label: str}} if Path(labeled).suffix == ".csv" else {}
        return cls.from_frame(read_table(labeled, **kw), label, features)

    from_csv = from_files

    def __len__(self) -> int:
        return len(self.frame)

    @cached_property
    def labeled(self) -> np.ndarray:
        lab = self.frame[self.label]
        return (lab.notna() & lab.astype(str).str.strip().ne("")).to_numpy()

    @cached_property
    def _encoded(self) -> tuple[np.ndarray, np.ndarray]:
        return np.unique(self.frame.loc[self.labeled, self.label].astype(str),
                         return_inverse=True)

    @property
    def classes(self) -> np.ndarray:
        return self._encoded[0]

    @property
    def y(self) -> np.ndarray:
        return self._encoded[1]

    @cached_property
    def X(self) -> np.ndarray:
        return self.frame[self.features].to_numpy(dtype=np.float32)

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(self.frame[mask].reset_index(drop=True), self.features, self.label)


## A named collection practice and the Z (Specs) it reads.
@dataclass
class Protocol:
    name: str
    z: list = field(default_factory=list)
    label: str = ""


## Plugins (e.g. pappi.geo) register named protocols here on import.
PROTOCOLS: dict[str, Protocol] = {}


def register_protocol(p: Protocol) -> Protocol:
    PROTOCOLS[p.name] = p
    return p


def get_protocol(p: str | Protocol) -> Protocol:
    if isinstance(p, Protocol):
        return p
    if p not in PROTOCOLS:
        raise KeyError(f"unknown protocol {p!r}; registered: {sorted(PROTOCOLS)} "
                       "(geo protocols register on `import pappi.geo`)")
    return PROTOCOLS[p]


## Adds each (name, fn) column unless present; returns the column name per spec.
def materialize(frame: pd.DataFrame, specs: list[Spec]) -> tuple[pd.DataFrame, list[str]]:
    names, copied = [], False
    for spec in specs:
        name, fn = (spec, None) if isinstance(spec, str) else spec
        if name not in frame.columns:
            if fn is None:
                raise ValueError(f"missing column {name!r}")
            frame, copied = (frame if copied else frame.copy()), True
            frame[name] = np.asarray(fn(frame), dtype=np.float64)
        names.append(name)
    return frame, names


## XGBoost trainer with the paper's settings (200 rounds, depth 6, eta 0.15).
def xgb_fitter(light: bool = False, **params) -> FitFn:
    def fit(X, y, seed):
        from xgboost import XGBClassifier

        kw = dict(n_estimators=LIGHT_ROUNDS if light else 200, max_depth=6, learning_rate=0.15,
                  tree_method="hist", verbosity=0, random_state=seed)
        return XGBClassifier(**(kw | params)).fit(X, y)

    return fit


## Default fold assignment (the paper's): a random fold per row, or, with groups, whole
## groups to folds by a random permutation mod k. e.g. groups [a, a, b] -> folds [2, 2, 0].
def kfold(n: int, groups: np.ndarray | None, seed: int, k: int = N_FOLDS) -> np.ndarray:
    return assign_folds(n, groups, np.random.default_rng(seed), k)


def assign_folds(n: int, groups: np.ndarray | None, rng: np.random.Generator,
                 k: int = N_FOLDS) -> np.ndarray:
    if groups is None:
        return rng.integers(0, k, size=n)
    _, inv = np.unique(groups, return_inverse=True)
    return (rng.permutation(inv.max() + 1) % k)[inv]


## A named way to split rows into folds: groups (a Spec) and an assign function
## (None: kfold with the audit's `folds`).
@dataclass(eq=False)
class Split:
    name: str = ""
    groups: Spec | None = None
    assign: FoldFn | None = None


## Percentile interval from `n` replicates: labeled rows resampled by `groups` (default:
## the correction split's groups, else rows), deployment rows by row; propensity refit.
## `arms` ("x", "z", "xz", "pi") and `seeds` limit it.
@dataclass(eq=False)
class Bootstrap:
    n: int = 200
    groups: object = None
    arms: tuple[str, ...] | None = None
    seeds: list[int] | None = None

    def covers(self, key: str, seed: int) -> bool:
        return ((self.arms is None or key in self.arms)
                and (self.seeds is None or seed in self.seeds))


@dataclass
class Report:
    frame: pd.DataFrame
    data: Dataset
    notes: list[str] = field(default_factory=list)
    metric: str = "macro-F1"
    ## Per seed, the correction split's fold models, whose mean score the PPI rows estimate
    ## (with `model`, just that model); call them on data.X (the rows audit kept).
    models: dict[int, list[Fitted]] = field(default_factory=dict)

    @property
    def n_labels(self) -> int:
        return int(self.data.labeled.sum())

    @property
    def n_deploy(self) -> int:
        return len(self.data) - self.n_labels

    ## Mean over seeds; the shown interval is the bootstrap one only when every
    ## seed has one, so it never describes a different set of seeds than the score.
    def summary(self) -> pd.DataFrame:
        cols = [c for c in ("score", "ci_lo", "ci_hi", "boot_lo", "boot_hi", "n_eff", "auc")
                if c in self.frame.columns]
        s = self.frame.groupby("estimate", sort=False)[cols].mean()
        if "boot_lo" in s.columns:
            has = self.frame.boot_lo.notna().groupby(self.frame.estimate, sort=False).all()
            s.loc[has, "ci_lo"], s.loc[has, "ci_hi"] = s.boot_lo[has], s.boot_hi[has]
        return s

    def rows(self) -> list[tuple[str, str, str, str]]:
        return [(name, f"{r.score:.2f}",
                 "--" if np.isnan(r.ci_lo) else f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]",
                 f"{r.n_eff:.0f}") for name, r in self.summary().iterrows()]

    def caption(self) -> str:
        return " · ".join([f"{self.n_labels} labeled", f"{self.n_deploy} deployment", *self.notes])

    def __rich__(self):
        from rich import box
        from rich.table import Table

        t = Table("estimate", self.metric, "95% CI", "n_eff", box=box.SIMPLE_HEAD,
                  caption=self.caption(), caption_justify="left")
        for c in t.columns[1:]:
            c.justify = "right"
        rows = self.rows()
        for i, row in enumerate(rows):
            t.add_row(*row, end_section=row[2] == "--" and i + 1 < len(rows)
                      and rows[i + 1][2] != "--")
        return t

    def __str__(self) -> str:
        rows = [("estimate", self.metric, "95% CI", "n_eff"), *self.rows()]
        w = max(len(r[0]) for r in rows) + 2
        lines = [f"{r[0]:<{w}}{r[1]:>8}   {r[2]:<12}{r[3]:>7}" for r in rows]
        return "\n".join([lines[0], "-" * len(lines[0]), *lines[1:], self.caption()])

    __repr__ = __str__


## Any object with predict_proba, or a function f(X) -> probabilities, is a model. Column j
## of its output must mean class j of the sorted labels; named columns are passed by name.
def _as_model(m, data: Dataset) -> Fitted:
    if not hasattr(m, "predict_proba"):
        if not callable(m):
            raise TypeError("model needs predict_proba(X) or must be a function f(X) -> "
                            "probabilities")
        m = SimpleNamespace(predict_proba=m)
    names = [str(c) for c in getattr(m, "classes_", data.classes)]
    if names not in (list(data.classes), [str(i) for i in range(len(data.classes))]):
        raise ValueError(f"model classes {names} != label classes {list(data.classes)}")
    cols = getattr(m, "feature_names_in_", None)
    if cols is None:
        return m
    cols = [str(c) for c in cols]
    if sorted(cols) != sorted(data.features):
        raise ValueError(f"model features {cols} != table features {data.features}")
    idx = [data.features.index(c) for c in cols]
    return SimpleNamespace(
        predict_proba=lambda X: m.predict_proba(pd.DataFrame(X[:, idx], columns=cols)))


## predict_proba as an array with one column per class 0..k-1.
def _proba(m, X: np.ndarray, k: int) -> np.ndarray:
    p = np.asarray(m.predict_proba(X), dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != k:
        raise ValueError(f"predict_proba must return (n, {k}) columns for classes 0..{k - 1}, "
                         f"got {p.shape}")
    if not (np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-3)):
        raise ValueError("predict_proba must return finite values in [0, 1] that sum to 1 "
                         "in each row")
    return p


## Held-out probabilities: for each fold, fit on the other rows and predict this one.
## With x_u, also each fold model's probabilities on x_u (folds, len(x_u), k) and the models.
def _oof(fit, X, y, ids: np.ndarray, seed: int, k: int, x_u=None):
    out, p_u, models = None, [], []
    for f in np.unique(ids):
        te = ids == f
        m = fit(X[~te], y[~te], seed)
        p = _proba(m, X[te], k)
        out = np.zeros((len(X), k), dtype=p.dtype) if out is None else out
        out[te] = p
        if x_u is not None:
            p_u.append(_proba(m, x_u, k))
            models.append(m)
    return out if x_u is None else (out, np.stack(p_u), models)


## in_domain: Splits scored on the labels (the CV rows). correction_split: held-out
## predictions for the PPI correction (default in_domain[0]). propensity_split: folds for
## the labeled-vs-deployment model over both row sets (default: correction groups on L,
## each deployment row alone). inclusion: known inclusion probability, a column or
## f(frame) -> array, adds a "known inclusion" row. interval: None (delta) or Bootstrap.
## model: a trained model audited as is, in place of fit_model; the labeled rows must be
## a test set it never saw, or the estimate comes out optimistic. folds: k of the default
## (kfold) splits; each fold model trains on (k-1)/k of L. n_jobs: processes
## for the seeds, or for the bootstrap with one seed; the same results.
def audit(
    data: Dataset | pd.DataFrame | str | Path,
    fit_model: FitFn | None = None,
    model: Fitted | None = None,
    protocols: list[str | Protocol] = (),
    in_domain: list[Split] | None = None,
    correction_split: Split | None = None,
    propensity_split: Split | None = None,
    inclusion=None,
    interval: Bootstrap | None = None,
    metric: Metric | None = None,
    fit_propensity: FitFn | None = None,
    seeds: int | list[int] = 1,
    light: bool = False,
    folds: int = N_FOLDS,
    n_jobs: int = 1,
    progress: bool = False,
) -> Report:
    if not isinstance(data, Dataset):
        data = (Dataset.from_frame(data) if isinstance(data, pd.DataFrame)
                else Dataset.from_files(data))
    in_domain = in_domain or [Split("random row split (CV)")]
    correction_split = correction_split or in_domain[0]
    if model is not None:
        if fit_model is not None:
            raise ValueError("give fit_model or model, not both")
        given = _as_model(model, data)

        ## The given model becomes a trainer that ignores its data: one fold scores all of
        ## L and U with the given model, never retrained.
        def fit_model(X, y, seed):
            return given

        correction_split = Split("test set (as reported)", correction_split.groups,
                                 lambda n, g, s: np.zeros(n, dtype=int))
        in_domain = [correction_split]
    cfg = dict(fit_model=xgb_fitter(light) if fit_model is None else fit_model,
               fit_propensity=(xgb_fitter(light, subsample=0.8, colsample_bytree=0.8)
                               if fit_propensity is None else fit_propensity),
               metric=MacroF1() if metric is None else metric,
               seeds=list(range(seeds)) if isinstance(seeds, int) else list(seeds),
               in_domain=in_domain, correction=correction_split, propensity=propensity_split,
               inclusion=inclusion, interval=interval, light=light,
               folds=folds, n_jobs=n_jobs)

    z_specs = [z for p in protocols for z in get_protocol(p).z]
    g_specs = [x.groups for x in (*in_domain, correction_split, propensity_split, interval)
               if x is not None and x.groups is not None]
    bar = _bar(progress)
    bar.start()
    try:
        frame, z_cols = data.frame, []
        task = bar.add_task("covariates", total=len(z_specs) + len(g_specs))
        for i, spec in enumerate(z_specs + g_specs):
            frame, (name,) = materialize(frame, [spec])
            bar.update(task, description=name)
            if i < len(z_specs) and name not in z_cols:
                z_cols.append(name)
            bar.advance(task)
        bar.remove_task(task)
        report = _fit(Dataset(frame, data.features, data.label), z_cols, bar, **cfg)
    finally:
        bar.stop()
    if model is not None:
        report.notes.append("given model; labels assumed an unseen test set")
    return report


## Column name of a Spec: "site" -> "site", ("block_5km", fn) -> "block_5km".
def _name(spec: Spec | None) -> str | None:
    return None if spec is None else spec if isinstance(spec, str) else spec[0]


## Stand-in progress bar for worker processes, which cannot share the real one.
class _Quiet:
    def update(self, *args, **kwargs):
        pass

    advance = update


def _bar(enabled: bool):
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    return Progress(SpinnerColumn(), TextColumn("{task.description:<24}"), BarColumn(),
                    MofNCompleteColumn(), TimeElapsedColumn(), transient=True,
                    disable=not enabled)


def _groups(frame, col, rows):
    return None if col is None else frame.loc[rows, col].to_numpy()


## Percentile interval of fn(il, iu, sd) over resampled label clusters and deployment rows.
## Draws run in order here, so n_jobs never changes results. Copies keep their cluster id,
## so a duplicate is never scored by a model that saw its twin.
def _bootstrap(n: int, n_jobs: int, clusters: np.ndarray, n_u: int, rng, fn,
               draw_seed: bool = True) -> tuple[float, float]:
    _, inv = np.unique(clusters, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    m = len(members)

    def draws():
        for _ in range(n):
            il = np.concatenate([members[j] for j in rng.integers(0, m, m)])
            iu = rng.integers(0, n_u, n_u)
            yield il, iu, int(rng.integers(1 << 31)) if draw_seed else None

    ## joblib caps each worker at cores / n_jobs threads; n_jobs=1 runs in this process.
    reps = Parallel(n_jobs=n_jobs)(delayed(fn)(*d) for d in draws())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


## Every number the report shows comes from here.
def _fit(data, z_cols, bar, fit_model, fit_propensity, metric, seeds, in_domain,
         correction, propensity, inclusion, interval, light, folds, n_jobs) -> Report:
    notes = []
    ok = (np.isfinite(data.X).all(axis=1)
          & np.isfinite(data.frame[z_cols].to_numpy(dtype=np.float64)).all(axis=1))
    if (~ok).any():
        notes.append(f"{int((~ok).sum())} rows dropped (non-finite X or Z)")
    u = np.flatnonzero(~data.labeled & ok)
    if light and len(u) > LIGHT_U:
        ok[np.random.default_rng(0).choice(u, len(u) - LIGHT_U, replace=False)] = False
        notes.append(f"light mode: {LIGHT_U:,} of {len(u):,} deployment rows")
    if not ok.all():
        data = data.subset(ok)

    f, lab, y, k = data.frame, data.labeled, data.y, len(data.classes)
    X, Z = data.X, f[z_cols].to_numpy(dtype=np.float32)
    n_l = int(lab.sum())
    n_u = len(lab) - n_l
    ## Rows ordered labeled-first (as from_frames builds them) need no reordering copies.
    order = slice(None) if lab[:n_l].all() else np.r_[np.flatnonzero(lab), np.flatnonzero(~lab)]
    sets = {"x": X} | ({"z": Z, "xz": np.hstack([X, Z])} if z_cols else {})
    sets = {a: m[order] for a, m in sets.items()}
    x_l, x_u = sets["x"][:n_l], sets["x"][n_l:]
    s = np.r_[np.ones(n_l), np.zeros(n_u)].astype(np.int64)
    g_corr = _groups(f, _name(correction.groups), lab)
    if propensity is None:
        codes = None if g_corr is None else pd.factorize(g_corr)[0]
        pg = None if codes is None else np.r_[codes, codes.max() + 1 + np.arange(n_u)]
    else:
        g = _name(propensity.groups)
        pg = None if g is None else pd.factorize(np.r_[_groups(f, g, lab), _groups(f, g, ~lab)])[0]
    kf = partial(kfold, k=folds)
    ## Propensity folds: the user's assign when given, else kfold.
    p_assign = (propensity.assign if propensity is not None else None) or kf
    w_pi = None
    if inclusion is not None:
        pi = np.asarray(f[inclusion] if isinstance(inclusion, str) else inclusion(f),
                        dtype=np.float64)[lab]
        if not (np.isfinite(pi).all() and (pi > 0).all()):
            raise ValueError("inclusion probabilities must be finite and > 0 on labeled rows")
        w_pi = inclusion_weights(pi)
    clusters = None
    if interval is not None:
        g = _groups(f, _name(interval.groups), lab)
        clusters = g if g is not None else (g_corr if g_corr is not None else np.arange(n_l))
        notes.append(f"bootstrap {interval.n}")

    splits = in_domain + ([] if correction in in_domain else [correction])
    per_seed = len(splits) + len(sets)
    task = bar.add_task("fits", total=len(seeds) * per_seed)

    ## Computed once here, so workers get each split's groups, not the whole frame.
    groups = [_groups(f, _name(sp.groups), lab) for sp in splits]

    def one_seed(seed, bar):
        rows = []
        for sp, g in zip(splits, groups):
            bar.update(task, description=f"seed {seed}: {sp.name or 'correction'}")
            ids = (sp.assign or kf)(n_l, g, seed)
            ## Cross-fit: each correction fold model also predicts U, so the imputation and
            ## the correction describe the same models.
            if sp is correction:
                p, p_u, models = _oof(fit_model, x_l, y, ids, seed, k, x_u)
                p_l = p
            else:
                p = _oof(fit_model, x_l, y, ids, seed, k)
            if sp in in_domain:
                rows.append(dict(seed=seed, key=sp.name, estimate=sp.name,
                                 score=metric.score(y, p.argmax(1), k), n_eff=float(n_l)))
            bar.advance(task)
        yl_hat, yu_hat = p_l.argmax(1), p_u.argmax(2)

        def est(w, il=slice(None), iu=slice(None)):
            return ppi_estimate(metric, p_l[il], yl_hat[il], y[il], w, p_u[:, iu], yu_hat[:, iu],
                                k)

        rng = np.random.default_rng(seed + 7)
        for key, F in sets.items():
            bar.update(task, description=f"seed {seed}: {ARMS[key]}")
            q = _oof(fit_propensity, F, s, p_assign(len(F), pg, seed), seed, 2)[:, 1]
            r = est(inverse_odds_weights(q[:n_l]))
            row = dict(seed=seed, key=key, estimate=ARMS[key], score=r["point"], ci_lo=r["ci_lo"],
                       ci_hi=r["ci_hi"], sigma=r["sigma"], n_eff=r["n_eff"], clipped=r["clipped"],
                       auc=float(roc_auc_score(s, q)), q_l_median=float(np.median(q[:n_l])),
                       q_u_median=float(np.median(q[n_l:])),
                       q_clip_share_l=float(np.mean(q[:n_l] <= _PI_CLIP_MIN)))
            if interval is not None and interval.covers(key, seed):
                def rep(il, iu, sd, F=F):
                    rr = np.r_[il, n_l + iu]
                    ## Copies keep their row or group id, so twins share a fold.
                    a = p_assign(len(rr), rr if pg is None else pg[rr], sd)
                    q_b = _oof(fit_propensity, F[rr], s[rr], a, sd, 2)[:, 1]
                    return est(inverse_odds_weights(q_b[:len(il)]), il, iu)["point"]

                row["boot_lo"], row["boot_hi"] = _bootstrap(
                    interval.n, boot_jobs, clusters, n_u, rng, rep)
            rows.append(row)
            bar.advance(task)

        if w_pi is not None:
            r = est(w_pi)
            row = dict(seed=seed, key="pi", estimate=ARMS["pi"], score=r["point"],
                       ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], sigma=r["sigma"], n_eff=r["n_eff"],
                       clipped=r["clipped"])
            if interval is not None and interval.covers("pi", seed):
                ## Serial: a known-weights replicate is too cheap to ship to a process.
                row["boot_lo"], row["boot_hi"] = _bootstrap(
                    interval.n, 1, clusters, n_u, rng,
                    lambda il, iu, sd: est(w_pi[il], il, iu)["point"], draw_seed=False)
            rows.append(row)
        return rows, models

    ## One level of processes: seeds when there are several, else bootstrap replicates.
    ## Seeds and replicates draw in a fixed order, so n_jobs never changes results.
    outer = min(n_jobs, len(seeds))
    boot_jobs = n_jobs if outer == 1 else 1
    out = []
    for r in Parallel(n_jobs=outer, return_as="generator")(
            delayed(one_seed)(seed, bar if outer == 1 else _Quiet()) for seed in seeds):
        out.append(r)
        if outer != 1:
            bar.advance(task, per_seed)
    rows = [row for seed_rows, _ in out for row in seed_rows]
    models = dict(zip(seeds, (m for _, m in out)))

    if any(row.get("clipped") for row in rows):
        notes.append("a corrected rate fell below 0; use --bootstrap for the interval")
    if len(seeds) > 1:
        notes.append(f"mean of {len(seeds)} seeds")
    return Report(pd.DataFrame(rows), data, notes, metric.name, models)
