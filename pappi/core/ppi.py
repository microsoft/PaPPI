# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Protocol as Contract

import numpy as np
from sklearn.metrics import f1_score

## The propensity clip bounds the largest weight.
_PI_CLIP_MIN = 1e-3


## Per-sample [tp_0, fp_0, fn_0, tp_1, ...] expected under the model's probabilities.
def _g_vector(p_hat: np.ndarray, y_hat: np.ndarray, K: int) -> np.ndarray:
    n = len(p_hat)
    out = np.zeros((n, 3 * K), dtype=np.float64)
    for k in range(K):
        pk = p_hat[:, k]
        eq = (y_hat == k).astype(np.float64)
        ne = 1.0 - eq
        out[:, 3 * k + 0] = pk * eq
        out[:, 3 * k + 1] = (1.0 - pk) * eq
        out[:, 3 * k + 2] = pk * ne
    return out


def _h_vector(y: np.ndarray, y_hat: np.ndarray, K: int) -> np.ndarray:
    n = len(y)
    out = np.zeros((n, 3 * K), dtype=np.float64)
    for k in range(K):
        is_k = (y == k).astype(np.float64)
        pr_k = (y_hat == k).astype(np.float64)
        out[:, 3 * k + 0] = is_k * pr_k
        out[:, 3 * k + 1] = (1.0 - is_k) * pr_k
        out[:, 3 * k + 2] = is_k * (1.0 - pr_k)
    return out


## Rates are clipped to >= 0 and F1 to [0, 1]; without it a collapsed n_eff can push
## a denominator through zero.
def _phi_macro_f1(mu: np.ndarray, K: int) -> float:
    eps = 1e-12
    mu = np.maximum(mu, 0.0)
    f1s = []
    for k in range(K):
        tp, fp, fn = mu[3 * k + 0], mu[3 * k + 1], mu[3 * k + 2]
        den = 2 * tp + fp + fn
        f1s.append(2 * tp / max(den, eps))
    return float(np.clip(np.mean(f1s), 0.0, 1.0))


def _phi_grad(mu: np.ndarray, K: int) -> np.ndarray:
    eps = 1e-12
    mu = np.maximum(mu, 0.0)
    grad = np.zeros(3 * K, dtype=np.float64)
    for k in range(K):
        tp, fp, fn = mu[3 * k + 0], mu[3 * k + 1], mu[3 * k + 2]
        den = max(2 * tp + fp + fn, eps)
        d2 = den * den
        grad[3 * k + 0] = (2.0 * (fp + fn)) / (K * d2)
        grad[3 * k + 1] = (-2.0 * tp) / (K * d2)
        grad[3 * k + 2] = (-2.0 * tp) / (K * d2)
    return grad


## A metric PPI can correct: a smooth function of per-row means. `observed`
## rows come from labels, `expected` rows from predicted probabilities, and
## `value`/`grad` map the mean vector to the metric and its gradient.
class Metric(Contract):
    name: str

    def observed(self, y: np.ndarray, y_hat: np.ndarray, k: int) -> np.ndarray: ...
    def expected(self, p: np.ndarray, y_hat: np.ndarray, k: int) -> np.ndarray: ...
    def value(self, mu: np.ndarray, k: int) -> float: ...
    def grad(self, mu: np.ndarray, k: int) -> np.ndarray: ...
    def score(self, y: np.ndarray, y_hat: np.ndarray, k: int) -> float: ...


class MacroF1:
    name = "macro-F1"

    def observed(self, y, y_hat, k):
        return _h_vector(y, y_hat, k)

    def expected(self, p, y_hat, k):
        return _g_vector(p, y_hat, k)

    def value(self, mu, k):
        return _phi_macro_f1(mu, k)

    def grad(self, mu, k):
        return _phi_grad(mu, k)

    def score(self, y, y_hat, k):
        return macro_f1(y, y_hat, k)


class Accuracy:
    name = "accuracy"

    def observed(self, y, y_hat, k):
        return (y == y_hat).astype(np.float64)[:, None]

    def expected(self, p, y_hat, k):
        return p[np.arange(len(p)), y_hat].astype(np.float64)[:, None]

    def value(self, mu, k):
        return float(np.clip(mu[0], 0.0, 1.0))

    def grad(self, mu, k):
        return np.ones(1)

    def score(self, y, y_hat, k):
        return float(np.mean(y == y_hat))


## Hájek-weighted mean of the rows of r, its covariance, and Kish's n_eff.
def _hajek(w: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w_sum = float(w.sum())
    mu = (w[:, None] * r).sum(axis=0) / w_sum
    c = r - mu
    V = (w[:, None] ** 2 * c).T @ c / (w_sum * w_sum)
    return mu, V, (w_sum * w_sum) / max(float((w * w).sum()), 1e-12)


## Metric at mu with a delta-method 95% interval from the covariance of mu.
def _delta(metric: Metric, mu: np.ndarray, Sigma: np.ndarray, K: int) -> dict:
    point = metric.value(mu, K)
    grad = metric.grad(mu, K)
    sigma = float(np.sqrt(max(float(grad @ Sigma @ grad), 0.0)))
    return {"point": point, "ci_lo": point - 1.96 * sigma, "ci_hi": point + 1.96 * sigma,
            "sigma": sigma, "clipped": bool((mu < 0).any())}


## Eq. (2) of the paper: imputation over U + Hájek-weighted
## residual over L, delta-method 95% CI. Cross-fit: p_hat_u (models, n_u, k) and
## y_hat_u (models, n_u) hold each fold model's predictions; U imputes their mean row.
def ppi_estimate(
    metric: Metric,
    p_hat_l: np.ndarray,
    y_hat_l: np.ndarray,
    y_l: np.ndarray,
    w_l: np.ndarray,
    p_hat_u: np.ndarray,
    y_hat_u: np.ndarray,
    k_classes: int,
) -> dict:
    K = k_classes
    g_l = metric.expected(p_hat_l, y_hat_l, K)
    h_l = metric.observed(y_l, y_hat_l, K)
    r_l = h_l - g_l
    g_u = (metric.expected(p_hat_u, y_hat_u, K) if p_hat_u.ndim == 2 else
           np.mean([metric.expected(p, yh, K) for p, yh in zip(p_hat_u, y_hat_u)], axis=0))

    N = len(g_u)
    mu_imp = g_u.mean(axis=0)
    centered_u = g_u - mu_imp
    S_imp = centered_u.T @ centered_u / N

    mu_res, V_res, n_eff = _hajek(w_l, r_l)
    mu_total = mu_imp + mu_res
    return _delta(metric, mu_total, S_imp / N + V_res, K) | {
        "rates": mu_total, "rectifier": mu_res, "imputation": mu_imp, "n_eff": n_eff}


## IWCV: the same weights on the observed rates over L alone, with no deployment term.
## Delta-method interval as in ppi_estimate, without the imputation variance.
def iwcv_estimate(metric: Metric, y_l: np.ndarray, y_hat_l: np.ndarray, w_l: np.ndarray,
                  k_classes: int) -> dict:
    mu, V, n_eff = _hajek(w_l, metric.observed(y_l, y_hat_l, k_classes))
    return _delta(metric, mu, V, k_classes) | {"n_eff": n_eff}


def ppi_macro_f1(p_hat_l, y_hat_l, y_l, w_l, p_hat_u, y_hat_u, k_classes: int) -> dict:
    return ppi_estimate(MacroF1(), p_hat_l, y_hat_l, y_l, w_l, p_hat_u, y_hat_u, k_classes)


## w = (1-q)/q is proportional to p_U/p_L; 1/q is not.
def inverse_odds_weights(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, _PI_CLIP_MIN, 1.0 - _PI_CLIP_MIN)
    return (1.0 - q) / q


## Known inclusion probabilities, needed only up to a constant (Hájek normalises);
## tiny values are floored at 1e-3 of the largest so one label cannot take all the weight.
def inclusion_weights(pi: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, dtype=np.float64)
    return 1.0 / np.maximum(pi, _PI_CLIP_MIN * pi.max())


## Fixed class list: sklearn otherwise averages over present classes only.
def macro_f1(y: np.ndarray, y_hat: np.ndarray, k_classes: int) -> float:
    return float(f1_score(y, y_hat, labels=list(range(k_classes)), average="macro",
                          zero_division=0.0))
