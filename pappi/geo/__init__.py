# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pappi.core.pipeline import Dataset, Protocol, register_protocol
from pappi.geo.covariates import (
    DistanceTo,
    DistToBuilding,
    DistToRoad,
    Elevation,
    SampleRaster,
    Slope,
    SlopeOf,
    point_covariate,
)
from pappi.geo.sources import (
    CACHE_DIR,
    CopernicusDEM,
    GeofabrikLayer,
    LocalRaster,
    LocalVector,
    cached,
)

## The paper's collection protocols; region and size read columns the user supplies.
for _p in (
    Protocol("roadside", [DistToRoad()], "Road access"),
    Protocol("settlement", [DistToBuilding()], "Settlement proximity"),
    Protocol("elevation", [Elevation()], "Elevation cut-off"),
    Protocol("slope", [Slope()], "Slope (walking time)"),
    Protocol("region", ["commune_id"], "Single-region pilot"),
    Protocol("size", ["log_area"], "Farm size floor"),
):
    register_protocol(_p)

DEFAULT_PROTOCOLS = ("roadside", "settlement", "elevation", "slope")


def is_geo(data: Dataset) -> bool:
    return {"lat", "lon"} <= set(data.frame.columns)


## audit() defaults for point data: the four protocols whose Z can be fetched.
def defaults(data: Dataset) -> dict:
    return {"protocols": DEFAULT_PROTOCOLS}


__all__ = [
    "CACHE_DIR", "DEFAULT_PROTOCOLS", "CopernicusDEM", "DistToBuilding", "DistToRoad",
    "DistanceTo", "Elevation", "GeofabrikLayer", "LocalRaster", "LocalVector", "point_covariate",
    "SampleRaster", "Slope", "SlopeOf", "cached", "defaults", "is_geo",
]
