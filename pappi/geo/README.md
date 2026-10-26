# pappi.geo

The geospatial plugin: `import pappi.geo` registers the paper's bias protocols.
Geo libraries load lazily (the `[geo]` extra).

- `sources.py`: acquire. `GeofabrikLayer`, `LocalVector` (vector); `CopernicusDEM`,
  `LocalRaster` (raster); `cached()` disk cache under `~/.cache/pappi`; `HttpFile`.
- `covariates.py`: derive. Factories returning `(name, fn)` specs: `point_covariate`,
  `DistanceTo(source)`, `SampleRaster(source)`, `SlopeOf(source)`;
  built-ins `DistToRoad`, `DistToBuilding`, `Elevation`, `Slope`.
- `__init__.py`: protocol registration, `DEFAULT_PROTOCOLS`, `defaults(data)`, `is_geo(data)`.
