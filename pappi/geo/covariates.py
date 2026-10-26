# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from pappi.geo.sources import CACHE_DIR, DRIVABLE, CopernicusDEM, GeofabrikLayer

## Each factory returns a Spec: (column name, fn(frame) -> array) over the lon/lat columns.
PointFn = Callable[[np.ndarray, np.ndarray, Path], np.ndarray]


def point_covariate(name: str, compute: PointFn, lat: str = "lat", lon: str = "lon",
                    cache_dir: str | None = None) -> tuple[str, Callable]:
    cache = Path(cache_dir).expanduser() if cache_dir else CACHE_DIR

    def fn(frame):
        return compute(frame[lon].to_numpy(dtype=np.float64),
                       frame[lat].to_numpy(dtype=np.float64), cache)

    return name, fn


## Metres from each point to the nearest feature of a vector source.
def DistanceTo(source, name: str, **kw):
    def compute(lon, lat, cache):
        import geopandas as gpd
        from shapely import STRtree

        geoms = source.fetch(_bbox(lon, lat, 0.05), cache)
        if len(geoms) == 0:
            return np.full(len(lon), np.nan)
        pts = _project(lon, lat)
        tgt = gpd.GeoSeries(geoms, crs="EPSG:4326").to_crs(pts.crs)
        return pts.values.distance(tgt.values[STRtree(tgt.values).nearest(pts.values)])

    return point_covariate(name, compute, **kw)


## Raster value at each point, read from a local UTM grid at `res` metres.
def SampleRaster(source, name: str, res: float = 30.0, **kw):
    def compute(lon, lat, cache):
        grid, row, col = _grid(source, lon.tobytes(), lat.tobytes(), cache, res)
        return grid[row, col]

    return point_covariate(name, compute, **kw)


## Slope (degrees) of a raster at each point, from the same local UTM grid.
def SlopeOf(source, name: str, res: float = 30.0, **kw):
    def compute(lon, lat, cache):
        grid, row, col = _grid(source, lon.tobytes(), lat.tobytes(), cache, res)
        dz_dy, dz_dx = np.gradient(grid)
        return np.degrees(np.arctan(np.hypot(dz_dx / res, dz_dy / res)))[row, col]

    return point_covariate(name, compute, **kw)


## Blocks for grouped CV: `km` squares on an equirectangular grid, or `deg` cells on the
## lon/lat grid (deg=0.045 is the paper's Kenya grid, ~5 km near the equator).
def SpatialBlocks(km: float = 5.0, deg: float | None = None, **kw):
    def compute(lon, lat, cache):
        if deg:
            return (lat // deg) * 100_000 + lon // deg
        x = lon * 111.320 * np.cos(np.deg2rad(np.nanmean(lat)))
        return np.floor(x / km) * 1e6 + np.floor(lat * 110.574 / km)

    return point_covariate(f"block_{deg:g}deg" if deg else f"block_{km:g}km", compute, **kw)


def DistToRoad(**kw):
    return DistanceTo(GeofabrikLayer("roads", DRIVABLE), "dist_to_road", **kw)


def DistToBuilding(**kw):
    return DistanceTo(GeofabrikLayer("buildings"), "dist_to_building", **kw)


def Elevation(**kw):
    return SampleRaster(CopernicusDEM(), "elevation", **kw)


def Slope(**kw):
    return SlopeOf(CopernicusDEM(), "slope", **kw)


def _bbox(lon, lat, pad: float) -> tuple[float, float, float, float]:
    return (float(np.nanmin(lon)) - pad, float(np.nanmin(lat)) - pad,
            float(np.nanmax(lon)) + pad, float(np.nanmax(lat)) + pad)


def _project(lon, lat):
    import geopandas as gpd

    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    return pts.to_crs(pts.estimate_utm_crs())


## One reprojection per (source, points): elevation and slope share it.
@lru_cache(maxsize=4)
def _grid(source, lon_b: bytes, lat_b: bytes, cache: Path, res: float):
    import rasterio
    from rasterio.warp import Resampling, reproject

    lon, lat = np.frombuffer(lon_b), np.frombuffer(lat_b)
    path = source.fetch(_bbox(lon, lat, 0.02), cache)
    pts = _project(lon, lat)
    xs, ys = pts.x.to_numpy(), pts.y.to_numpy()
    x0, y1 = xs.min() - 5 * res, ys.max() + 5 * res
    w = int(np.ceil((xs.max() + 5 * res - x0) / res))
    h = int(np.ceil((y1 - (ys.min() - 5 * res)) / res))
    grid = np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=grid,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=rasterio.transform.from_origin(x0, y1, res, res),
                  dst_crs=pts.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    col = np.clip(((xs - x0) / res).astype(int), 0, w - 1)
    row = np.clip(((y1 - ys) / res).astype(int), 0, h - 1)
    return grid, row, col
