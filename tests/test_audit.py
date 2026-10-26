# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from pappi import Protocol, Split, audit
from pappi.cli import main
from pappi.demo import synthetic_campaign, true_score

ROADSIDE = [Protocol("roadside", ["dist_to_road"])]


## End to end on five draws of the roadside demo: random CV is ~0.10 too high, and the
## PPI estimate is off by ~0.04 on average (macOS). Single draws vary too much to test
## one at a time; test_ppi.py checks the interval coverage of the estimator itself.
def test_audit_beats_random_cv():
    errs = []
    for seed in range(5):
        frame = synthetic_campaign(seed=seed)
        r = audit(frame, protocols=ROADSIDE, in_domain=[Split("cv")], light=True)
        s = r.frame.set_index("key").score
        truth = true_score(frame, light=True)
        errs.append((s["cv"] - truth, abs(s["x"] - truth)))
    cv, ppi = np.mean(errs, axis=0)
    assert cv > 0.05
    assert ppi < 0.7 * cv


def test_cli_writes_per_seed_rows(tmp_path):
    csv, out = tmp_path / "campaign.csv", tmp_path / "rows.csv"
    synthetic_campaign(n_labeled=300, n_deploy=600).to_csv(csv, index=False)
    assert main(["audit", str(csv), "--light", "--seeds", "2", "--out", str(out)]) == 0
    rows = pd.read_csv(out)
    assert len(rows) == 4 and "x" in set(rows.key)
