# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from pappi import Accuracy, Bootstrap, MacroF1, Protocol, Split, audit
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
        truth = true_score(frame, r.models)
        errs.append((s["cv"] - truth, abs(s["x"] - truth)))
    cv, ppi = np.mean(errs, axis=0)
    assert cv > 0.05
    assert ppi < 0.7 * cv


def test_cli_writes_rows(tmp_path):
    csv, out = tmp_path / "campaign.csv", tmp_path / "rows.csv"
    synthetic_campaign(n_labeled=300, n_deploy=600).to_csv(csv, index=False)
    assert main(["audit", str(csv), "--light", "--out", str(out)]) == 0
    rows = pd.read_csv(out)
    assert list(rows.key) == ["random row split (CV)", "x"]


## A trainer whose output swings with its training rows: it predicts its first label. The
## interval must still cover the models it reports on (a refit model once scored 0.10 here).
def test_crossfit_covers_the_models_it_scores():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(900, int), np.ones(100, int)]
    rng.shuffle(y)
    y_u = np.r_[np.zeros(900, int), np.ones(100, int)]
    frame = pd.DataFrame({"x_0": rng.normal(size=2000),
                          "label": np.r_[np.array(["A", "B"])[y], [""] * 1000]})

    class First:
        def __init__(self, c):
            self.c = c

        def predict_proba(self, X):
            return np.eye(2)[np.full(len(X), self.c)]

    r = audit(frame, fit_model=lambda X, y: First(int(y[0])), metric=Accuracy(),
              inclusion=lambda f: np.ones(len(f)))
    row = r.frame.set_index("key").loc["pi"]
    truth = np.mean([np.mean(m.predict_proba(np.zeros((1000, 1))).argmax(1) == y_u)
                     for m in r.models])
    assert row.ci_lo <= truth <= row.ci_hi


@pytest.mark.parametrize("metric, truth", [
    (Accuracy(), 0.5),
    (MacroF1(), (8 / 13 + 2 / 7) / 2),
])
@pytest.mark.parametrize("clusters", [None, "fold", "cluster", "mixed"])
def test_unequal_correction_folds_keep_model_target(metric, truth, clusters):
    fold_ids = np.r_[np.full(900, -7), np.full(100, 13)]
    frame = pd.DataFrame({
        "x_0": np.zeros(2000),
        "label": np.r_[np.tile(["A", "A", "A", "A", "B"], 200), [""] * 1000],
        "fold": np.r_[fold_ids, np.zeros(1000)],
        "cluster": np.arange(2000) % 50,
        "mixed": np.r_[np.zeros(899), np.ones(101), np.zeros(1000)],
        "p_selected": np.r_[np.ones(900), np.full(100, 0.2), np.ones(1000)],
    })

    class Constant:
        def __init__(self, probabilities):
            self.probabilities = probabilities

        def predict_proba(self, X):
            return np.tile(self.probabilities, (len(X), 1))

    ## The models always predict A and B, respectively, on an 80/20 population.
    def trainer(X, y):
        return Constant([0.9, 0.1] if len(y) == 100 else [0.4, 0.6])

    split = Split("unequal", groups="fold" if clusters == "fold" else None,
                  assign=lambda n, g: fold_ids)
    report = audit(
        frame, fit_model=trainer, fit_propensity=lambda X, y: Constant([0.5, 0.5]),
        in_domain=[split], inclusion="p_selected", metric=metric, iwcv=True,
        interval=Bootstrap(30, groups=clusters, arms=("x", "pi")),
    )
    rows = report.frame.set_index("key")
    for key in ("x", "pi", "iwcv_x", "iwcv_pi"):
        row = rows.loc[key]
        assert row.score == pytest.approx(truth)
        assert row.ci_lo <= truth <= row.ci_hi
        if isinstance(metric, Accuracy):
            assert row.sigma == pytest.approx(np.sqrt(0.8 * 0.2 / 4 * (1 / 900 + 1 / 100)))
        assert row.boot_lo - 1e-12 <= truth <= row.boot_hi + 1e-12


def test_custom_propensity_assign_without_groups_is_used():
    calls = []

    def assign(n, groups):
        calls.append(n)
        return np.arange(n) % 5

    frame = synthetic_campaign(n_labeled=200, n_deploy=400)
    audit(frame, propensity_split=Split("p", assign=assign), light=True)
    assert calls


def test_bad_probabilities_raise():
    logits = lambda X: np.tile([3.0, -3.0, 0.0], (len(X), 1))  # noqa: E731
    with pytest.raises(ValueError, match="sum to 1"):
        audit(synthetic_campaign(n_labeled=200, n_deploy=400), model=logits, light=True)


## A protocol chosen on lat/lon data replaces the geo default ones; dist_to_road is in the
## table, so nothing is downloaded. Parquet input needs pyarrow.
def test_cli_protocols_override_geo_defaults(tmp_path):
    frame = synthetic_campaign(n_labeled=300, n_deploy=600)
    rng = np.random.default_rng(0)
    frame["lat"], frame["lon"] = rng.uniform(0, 0.3, len(frame)), rng.uniform(36, 36.3, len(frame))
    data, out = tmp_path / "campaign.parquet", tmp_path / "rows.csv"
    frame.to_parquet(data)
    assert main(["audit", str(data), "--protocols", "roadside", "--light", "--out", str(out)]) == 0
    assert list(pd.read_csv(out).key) == ["random row split (CV)", "x", "z", "xz"]


def test_local_vector_in_projected_crs(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    from pappi.geo import LocalVector

    road = gpd.GeoSeries([LineString([(35.99, -1.0), (36.01, -1.0)])], crs=4326).to_crs(3857)
    gpd.GeoDataFrame(geometry=road).to_file(tmp_path / "roads.gpkg")
    assert len(LocalVector(str(tmp_path / "roads.gpkg")).fetch((35.98, -1.02, 36.02, -0.98),
                                                               tmp_path)) == 1


def test_http_file_rejects_ignored_range():
    from types import SimpleNamespace

    from pappi.geo.sources import HttpFile

    def reply(code, body=b"", headers=None):
        return SimpleNamespace(status_code=code, content=body, headers=headers or {}, url="u",
                               raise_for_status=lambda: None)

    session = SimpleNamespace(
        headers={},
        head=lambda *a, **k: reply(200, headers={"content-length": "10", "accept-ranges": "bytes"}),
        get=lambda *a, **k: reply(200, b"0123456789"))
    f = HttpFile("u", session=session)
    f.seek(5)
    with pytest.raises(OSError, match="HTTP 200"):
        f.read(2)
