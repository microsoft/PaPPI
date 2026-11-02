# PaPPI

PaPPI (Protocol-aware Prediction-Powered Inference) estimates a crop map's macro-F1 or accuracy across its deployment area, with a confidence interval. It uses biased field labels and unlabeled deployment points. Crop maps are usually evaluated on labels collected near roads or in a few survey areas, so their reported scores can overstate performance across the area they cover.

![audit output](docs/audit_output.svg)

## Setup

PaPPI requires Python 3.10 or newer.

```console
$ pip install "pappi[geo] @ git+https://github.com/microsoft/PaPPI"
```

The `[geo]` extra adds the OpenStreetMap and Copernicus DEM fetchers. Omit it if your table already contains its bias covariates.

The following commands install the exact dependency versions tested for this release:

```console
$ git clone https://github.com/microsoft/PaPPI && cd PaPPI
$ pip install -r requirements.txt
$ pip install --no-deps .
```

## Run

Audit your campaign with the columns described under [Input](#input):

```console
$ pappi audit my_campaign.csv
```

The synthetic roadside example runs in about 10 seconds without downloads:

```console
$ python examples/quickstart.py
```

Random cross-validation (CV) scores the near-road labels. The PPI rows estimate performance on the deployment rows, and the last line prints the deployment score calculated from hidden labels.

Audit a trained model with labeled rows that it has never seen:

```console
$ pappi audit test_and_deploy.csv --model my_model.joblib
```

## Input

Each row represents one sample. Rows with a `label` form the labeled set. Rows with an empty `label` must be a uniform sample of the deployment population. `--label` changes the label column name, and `--features` changes the feature-column prefix.

| Column | Meaning |
|---|---|
| `label` | The class name. Leave it empty for deployment rows. |
| `x_*` | Numeric model features. Encode categorical features first. PaPPI drops rows with missing feature values and reports the count in the caption. |
| `lat`, `lon` | Point coordinates. The CLI applies the four geo defaults when these exact column names are present and no protocols are specified. |
| Z columns, such as `dist_to_road` or `elevation` | Covariates that influenced labeling, selected by a protocol. PaPPI fetches missing geo covariates from OpenStreetMap or Copernicus DEM. |

Integer labels can become `3.0` when you use a separate deployment file. A trained `model` with class `3` will then be rejected. Keep the rows in one CSV or store the labels as text.

## Python

Importing `pappi.geo` registers the geo protocols. `geo.defaults(data)` selects the same protocols that the CLI uses for tables with `lat` and `lon`.

```python
from sklearn.linear_model import LogisticRegression
from pappi import Accuracy, Bootstrap, Dataset, Protocol, audit
import pappi.geo as geo

data = Dataset.from_csv("my_campaign.csv")
report = audit(data, **geo.defaults(data))


def fit_logreg(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


report = audit(
    Dataset.from_files("labels.parquet", "deploy.parquet"),
    fit_model=fit_logreg,
    protocols=[Protocol("referral", z=["distance_to_clinic"])],
    inclusion="p_selected",
    interval=Bootstrap(200),
    metric=Accuracy(),
)
report.frame
```

### Data and protocols

`Dataset.from_csv` and `Dataset.from_files(labeled, deploy=None)` read CSV, Parquet or Feather files. `Dataset.from_frame` accepts a pandas DataFrame.

`protocols` accepts `Protocol(name, z)` objects or registered names. `import pappi.geo` registers `roadside`, `settlement`, `elevation`, `slope`, `region` and `size`. The geo defaults are the four fetchable protocols: `roadside`, `settlement`, `elevation` and `slope`.

Each Z covariate is a column name or a `(name, fn)` pair, where `fn(frame)` returns an array. PaPPI calls the function only when the column is missing.

### Models and metrics

`fit_model` and `fit_propensity` accept functions `fit(X, y)` that return objects with `predict_proba(X)`. PaPPI calls each trainer once per fold. Probability columns must correspond to classes `0..K-1`. Both default trainers use XGBoost with the paper's settings.

`model` accepts a trained object with `predict_proba(X)` or a function `f(X)` that returns probabilities. PaPPI never retrains it and replaces the CV rows with one "test set (as reported)" row. Its labeled rows must be an unseen test set; otherwise the estimate is optimistic. PaPPI cannot check this.

Probability columns must follow the sorted labels. PaPPI checks their order when the model has `classes_` and passes named features by name when the model was fitted on them.

```python
report = audit(
    Dataset.from_files("test.parquet", "deploy.parquet"),
    model=my_model,
    protocols=[Protocol("referral", z=["distance_to_clinic"])],
)
```

`metric` accepts `MacroF1()` (the default), `Accuracy()`, or an object with `observed`, `expected`, `value`, `grad` and `score` methods.

`report.frame` contains one row per estimate. `report.models` contains the correction split's fold models, or `[your object]` when you pass `model`. PaPPI wraps functions and models fitted on named columns. Call the returned models on `report.data.X`; `report.data` contains only the rows kept by the audit.

### Folds

`Split(name, groups, assign)` defines a fold split. `groups` is a column name or `(name, fn)` pair, and rows in the same group share a fold. A custom `assign(n, groups)` returns one fold ID per row. The default `kfold` assigns rows, or whole groups, to random folds.

| Option | Purpose | Default |
|---|---|---|
| `in_domain` | These splits produce the CV rows. | One random row split. |
| `correction_split` | This split supplies held-out predictions for the PPI correction. | `in_domain[0]`. |
| `propensity_split` | This split trains the labeled-versus-deployment model across both row sets. | The correction groups on labeled rows, with each deployment row in its own group. |

Spatially grouped audits need the same groups in `propensity_split` (Python only). Otherwise deployment neighbors of a held-out block remain in training, which can distort the propensity model and produce `auc` below 0.5.

`folds=5` sets the number of folds for every default split. Each fold model trains on `(k-1)/k` of the labels. A custom `assign` controls its own number of folds.

### Weights and intervals

`inclusion` supplies known inclusion probabilities as a column name or a function `frame -> array`. The probabilities need only be proportional to the true inclusion probabilities. This option adds a "known inclusion" row.

The default confidence interval uses the faster delta method, which treats labels as independent and weights as fixed. `interval=Bootstrap(n, groups, arms)` gives percentile intervals by resampling labels or label clusters and refitting the propensity model for each replicate. `groups` selects label clusters; by default, the bootstrap uses the correction split's groups, or individual rows when there are no groups.

`arms` limits which estimates receive bootstrap intervals. For example, `("x",)` covers both the PPI row for X and its IWCV row, `iwcv_x`. PaPPI stores the bounds as `boot_lo` and `boot_hi` and displays them in the table. `n_jobs` sets the number of parallel bootstrap processes without changing the results.

`iwcv=True` adds an importance-weighted CV (IWCV) row after each PPI row, with key `iwcv_<arm>`. It scores the held-out labels with the same weights but uses no deployment predictions. IWCV is nearly as accurate as PPI in the paper, with slightly higher per-run error.

`light=True` samples at most 2,000 deployment rows and uses 50 boosting rounds. The caption reports the sample count and original count. `progress=True` shows a progress bar; the CLI enables it by default.

## CLI configuration

The CLI exposes `--fit-model pkg.module:fit`, `--fit-propensity pkg.module:fit`, `--model pkg.module:obj`, `--folds K`, `--jobs J`, `--inclusion COL`, `--bootstrap N`, `--metric accuracy` and `--iwcv`. Use `PYTHONPATH=. pappi audit ...` to load a module from the current folder.

`--model` also accepts a `.pkl` or `.joblib` file. Loading these files runs their code, so use trusted files only.

`--config` reads YAML with the same options, plus custom protocols and splits. `<key>_params` calls `fit_model`, `model` or `fit_propensity` as a factory with the supplied arguments.

```yaml
label: band
model: my_model.joblib
protocols:
  - {name: enrolment, z: [formal_sector, km_to_office]}
bootstrap: 500
jobs: 8
```

## Geo covariates

Each geo covariate returns a `(name, fn)` pair. Its source supplies vector or raster data, and its function computes values at the sample points:

```python
geo.DistanceTo(geo.LocalVector("my_roads.gpkg"), "dist_to_road")
geo.SampleRaster(geo.LocalRaster("my_dem.tif"), "elevation")
geo.SlopeOf(geo.CopernicusDEM(), "slope")
```

The available sources are `GeofabrikLayer(layer, fclass)`, `LocalVector(path)`, `CopernicusDEM()` and `LocalRaster(path)`. PaPPI caches downloads under `~/.cache/pappi`.

## Output

CV rows report in-domain cross-validation on the labels. A supplied `model` produces one "test set (as reported)" row instead. PPI rows estimate deployment performance with a propensity model fitted on X, Z or both. `n_eff` is the effective number of labels supporting the estimate. IWCV rows reweight the held-out labels with the same weights and omit deployment predictions.

The caption reports row counts and notes such as `light mode: 2,000 of 5,000 deployment rows`. `report.frame` also includes the propensity `auc`, propensity medians and `q_clip_share_l`, the share of labels whose propensity reaches the 1e-3 clip.

PPI estimates are not usable when `n_eff` is below a few dozen or `auc` is near 1. An AUC near 1 means the labeled and deployment rows barely overlap, and the interval is not reliable.

## Assumptions

- Labeling must be independent of the class after conditioning on the propensity covariates (X, Z or both). Each PPI or IWCV row makes this assumption for its own covariates.
- Every kind of deployment row must have some chance of being labeled. Weights hit the clip and `n_eff` falls where this assumption fails.
- Deployment rows are a uniform sample of the population the estimate is about.
- Held-out labels must be as new to each fold model as deployment rows. Split clustered labels, such as several rows per photo, field or transect, with `Split(name, groups=...)`. The bootstrap then resamples whole clusters. The default row split can put near-duplicates on both sides and bias the estimate.
- With `model`, the labeled rows were not used to train it.
- With `fit_model`, PPI estimates the mean deployment score of the correction split's fold models, each trained on `(k-1)/k` of the labels. Each model predicts the deployment rows and is corrected on labels it did not see. A model refitted on all labels usually scores a little higher.
- Labels are classes; the metrics are macro-F1 and accuracy.

## Tests

```console
$ pip install pytest && pytest
```

## Data and licenses

The repository contains no data. `pappi.demo` generates the demo campaign. PaPPI with `[geo]` downloads OpenStreetMap (© OpenStreetMap contributors, ODbL) and Copernicus DEM GLO-30 (© DLR e.V., © Airbus Defence and Space GmbH, provided under COPERNICUS by the EU and ESA) on your machine. The code is MIT-licensed ([LICENSE](LICENSE)). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists third-party packages and data attributions.

PaPPI sends the bounding box of your points to the Planetary Computer STAC API to find DEM tiles. It never sends labels, features or the points themselves. It then reads the DEM tiles and Geofabrik extracts that cover the box and caches downloads under `~/.cache/pappi`. PaPPI makes no requests when you supply the Z columns or use `LocalVector` and `LocalRaster`.

## Citation

```bibtex
@misc{zaytar2026pappi,
  title  = {Crop Mapping Evaluation under Sampling Bias},
  author = {Zaytar, Akram and Ochwang'i, Kemunto and Robinson, Caleb and Tadesse, Girmaw A.
            and Hacheme, Gilles Q. and Dodhia, Rahul and Becker-Reshef, Inbal
            and Lavista Ferres, Juan M.},
  year   = {2026},
}
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Authors are listed in [AUTHORS.md](AUTHORS.md). See [SUPPORT.md](SUPPORT.md) for help and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-party's policies.
