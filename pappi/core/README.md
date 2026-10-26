# pappi.core

The generic audit engine; usage in the top-level `README.md`. Imports nothing dataset-specific: no geo code
and no research modules. Plugins such as `pappi.geo` build on it.

- `ppi.py`: the `Metric` contract, `MacroF1`, `Accuracy`, `ppi_estimate` (IPW-PPI with a
  delta-method interval), inverse-odds and known-inclusion weights.
- `pipeline.py`: `Dataset`, `Protocol` and its registry, `Split`, `Bootstrap`, function contracts
  (`FitFn` = `fit(X, y) -> Fitted`, `FoldFn` = `assign(n, groups) -> fold ids`,
  `Spec` = column name or `(name, fn(frame))`), `kfold` (random fold
  assignment), `xgb_fitter`, `materialize`, `audit`, `Report`.
