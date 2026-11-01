# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from pappi.core.ppi import Accuracy, MacroF1, inclusion_weights, macro_f1, ppi_estimate


## The estimator alone, with known inclusion weights. The model is calibrated on near rows
## and overconfident on far rows, which the labels undersample 5x; e.g. it says 0.9 where
## the truth is 0.56. So the unweighted estimate is biased high and the weights must fix it.
def test_ppi_interval_covers_truth_with_known_weights():
    rng = np.random.default_rng(0)
    N, K = 200_000, 3
    far = rng.random(N) < 0.5
    p = rng.dirichlet(np.full(K, 0.5), N)
    q = np.where(far[:, None], 0.4 * p + 0.6 / K, p)
    y = (rng.random(N)[:, None] > np.cumsum(q, 1)).sum(1)
    y_hat = p.argmax(1)
    truth = macro_f1(y, y_hat, K)
    pi = np.where(far, 0.2, 1.0)

    def estimate(w_fn, rng):
        lab = rng.choice(N, 1000, p=pi / pi.sum())
        dep = rng.choice(N, 5000)
        r = ppi_estimate(MacroF1(), p[lab], y_hat[lab], y[lab], w_fn(pi[lab]),
                         p[dep], y_hat[dep], K)
        return r["point"], r["ci_lo"] <= truth <= r["ci_hi"]

    weighted = np.array([estimate(inclusion_weights, rng) for _ in range(300)])
    unweighted = np.array([estimate(np.ones_like, rng) for _ in range(300)])
    assert 0.90 <= weighted[:, 1].mean() <= 0.99
    assert abs(weighted[:, 0].mean() - truth) < 0.005
    assert unweighted[:, 1].mean() < 0.5


def test_crossfit_estimate_requires_correction_fold_ids():
    p = np.tile([0.9, 0.1], (2, 1))
    p_u = np.stack([p, p])
    with pytest.raises(ValueError, match="fold_l"):
        ppi_estimate(Accuracy(), p, p.argmax(1), np.array([0, 1]), np.ones(2),
                     p_u, p_u.argmax(2), 2)
