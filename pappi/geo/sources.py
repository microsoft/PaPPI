# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "pappi"
GEOFABRIK = "https://download.geofabrik.de"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LAYER_STEMS = {"roads": "gis_osm_roads_free_1", "buildings": "gis_osm_buildings_a_free_1"}
DRIVABLE = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "track", "track_grade1",
    "track_grade2", "track_grade3", "track_grade4", "track_grade5", "road",
)


## Build `path` once: write to `<path>.part`, then rename, so a crash never
## leaves a truncated file that later runs would trust.
def cached(path: Path, build) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    build(tmp)
    tmp.replace(path)
    return path


## Vector sources: fetch(bbox, cache) -> EPSG:4326 GeoSeries of features in bbox.
@dataclass(frozen=True)
class GeofabrikLayer:
    layer: str
    fclass: tuple[str, ...] | None = None

    def fetch(self, bbox, cache: Path):
        import geopandas as gpd
        import pandas as pd

        parts = []
        for ext_id, url in _extracts_for_bbox(bbox, cache):
            g = gpd.read_file(_extract_layer(url, self.layer, cache / "geofabrik" / f"_{ext_id}"),
                              bbox=bbox)
            if self.fclass and "fclass" in g.columns:
                g = g[g["fclass"].isin(self.fclass)]
            if len(g):
                parts.append(g.to_crs("EPSG:4326").geometry)
        return gpd.GeoSeries(pd.concat(parts, ignore_index=True) if parts else [], crs="EPSG:4326")


@dataclass(frozen=True)
class LocalVector:
    path: str

    def fetch(self, bbox, cache: Path):
        import geopandas as gpd

        return gpd.read_file(self.path, bbox=bbox).to_crs("EPSG:4326").geometry


## Raster sources: fetch(bbox, cache) -> path of a GeoTIFF covering bbox.
@dataclass(frozen=True)
class CopernicusDEM:
    collection: str = "cop-dem-glo-30"

    def fetch(self, bbox, cache: Path) -> Path:
        key = hashlib.md5(repr(np.round(bbox, 4)).encode()).hexdigest()[:12]
        name = key if self.collection == "cop-dem-glo-30" else f"{self.collection}_{key}"
        return cached(cache / "dem" / f"{name}.tif", lambda tmp: self._build(bbox, tmp))

    def _build(self, bbox, out: Path) -> None:
        import planetary_computer
        import pystac_client
        from rasterio.merge import merge

        cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        hrefs = [i.assets["data"].href
                 for i in cat.search(collections=[self.collection], bbox=bbox).items()]
        if not hrefs:
            raise ValueError(f"no {self.collection} tiles intersect {bbox}")
        merge(hrefs, bounds=bbox, dst_path=out)


@dataclass(frozen=True)
class LocalRaster:
    path: str

    def fetch(self, bbox, cache: Path) -> Path:
        return Path(self.path)


## Smallest Geofabrik shapefile extracts whose union covers `bbox`.
@lru_cache(maxsize=16)
def _extracts_for_bbox(bbox, cache: Path) -> list[tuple[str, str]]:
    import geopandas as gpd
    import requests
    from shapely.geometry import box

    def download(tmp: Path) -> None:
        r = requests.get(f"{GEOFABRIK}/index-v1.json", timeout=180)
        r.raise_for_status()
        tmp.write_bytes(r.content)

    idx = gpd.read_file(cached(cache / "geofabrik" / "index-v1.json", download))
    idx["shp_url"] = [(json.loads(u) if isinstance(u, str) else (u or {})).get("shp")
                      for u in idx.get("urls", [None] * len(idx))]
    g = box(*bbox)
    hit = idx[idx["shp_url"].notna() & idx.geometry.intersects(g)]
    if len(hit) == 0:
        raise FileNotFoundError(f"no Geofabrik shapefile extract touches {bbox}")
    b = hit.geometry.bounds
    hit = hit.assign(_a=(b.maxx - b.minx) * (b.maxy - b.miny)).sort_values("_a")
    covering = hit[hit.geometry.contains(g)]
    if len(covering):
        return [(str(covering.iloc[0]["id"]), str(covering.iloc[0]["shp_url"]))]
    chosen, remaining = [], g
    for row in hit.itertuples():
        if remaining.is_empty:
            break
        if row.geometry.intersects(remaining):
            chosen.append((str(row.id), str(row.shp_url)))
            remaining = remaining.difference(row.geometry)
    return chosen


def _extract_layer(url: str, layer: str, out_dir: Path) -> Path:
    stem = LAYER_STEMS[layer]
    shp = out_dir / f"{stem}.shp"
    if shp.exists():
        return shp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s from %s", stem, url)
    zf = zipfile.ZipFile(HttpFile(url))
    names = [n for n in zf.namelist() if Path(n).stem == stem]
    if not names:
        raise FileNotFoundError(f"{stem} not in {url}")
    for name in sorted(names, key=lambda n: n.endswith(".shp")):  ## .shp last: it marks completion
        def copy(tmp: Path, name=name) -> None:
            with zf.open(name) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)

        cached(out_dir / Path(name).name, copy)
    return shp


## Seekable file over HTTP range requests; a short read
## silently truncates a shapefile.
class HttpFile(io.RawIOBase):
    def __init__(self, url: str, session=None):
        import requests

        self.s = session or requests.Session()
        ## Byte offsets index the stored file; gzip transfer encoding would shift them.
        self.s.headers["Accept-Encoding"] = "identity"
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        if r.headers.get("accept-ranges", "").lower() != "bytes":
            logger.warning("%s does not advertise range support", url)
        self.url = r.url
        self.size = int(r.headers["content-length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        out = bytearray()
        while len(out) < end - self.pos + 1:
            lo = self.pos + len(out)
            hi = min(lo + (32 << 20), end + 1) - 1
            for attempt in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={lo}-{hi}"},
                                   timeout=300)
                    r.raise_for_status()
                    out += r.content
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    logger.warning("range %d-%d failed (%s), retrying", lo, hi, e)
        self.pos += len(out)
        return bytes(out)
