# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pandas as pd

from pappi.core.pipeline import Dataset
from pappi.core.ppi import macro_f1

CROPS = np.array(["maize", "beans", "sorghum"])


## A roadside campaign with known truth: rows over 1 km from a road are harder to classify,
## and the campaign labels them 5x less often. Deployment rows keep their hidden label in
## `true_label`; their `label` is empty.
def synthetic_campaign(n_labeled: int = 1000, n_deploy: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 10 * n_labeled + n_deploy
    X = rng.normal(size=(n, 4))
    dist_to_road = np.exp(X[:, 0])
    noise = np.where(dist_to_road > 1, 2.0, 0.3)
    y = (X[:, 1:4] + noise[:, None] * rng.normal(size=(n, 3))).argmax(1)
    u = rng.choice(n, n_deploy, replace=False)
    rest = np.setdiff1d(np.arange(n), u)
    p = np.where(dist_to_road[rest] > 1, 0.2, 1.0)
    lab = rng.choice(rest, n_labeled, replace=False, p=p / p.sum())
    rows = np.r_[lab, u]
    frame = pd.DataFrame(X[rows], columns=[f"x_{i}" for i in range(4)])
    frame["dist_to_road"] = dist_to_road[rows]
    frame["true_label"] = CROPS[y[rows]]
    frame["label"] = np.where(np.arange(len(rows)) < n_labeled, frame["true_label"], "")
    return frame


## Macro-F1 on the hidden deployment labels of the models audit() scores
## (report.models[seed]), pooled: the score of their mean TP/FP/FN rates.
def true_score(frame: pd.DataFrame, models: list) -> float:
    data = Dataset.from_frame(frame)
    lab = data.labeled
    y_u = np.searchsorted(data.classes, frame.loc[~lab, "true_label"].astype(str))
    y_hat = np.concatenate([m.predict_proba(data.X[~lab]).argmax(1) for m in models])
    return macro_f1(np.tile(y_u, len(models)), y_hat, len(data.classes))
