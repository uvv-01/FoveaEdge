"""Spatial region representation for FoveaEdge.

Provides Region, RegionSet, and geometry utilities for spatial operations.
"""

from foveaedge.regions.region import Region
from foveaedge.regions.region_set import RegionSet
from foveaedge.regions.geometry import (
    intersection_area,
    iou,
    overlap_ratio,
    contains_point,
    contains_region,
    clip_region,
)

__all__ = [
    "Region",
    "RegionSet",
    "intersection_area",
    "iou",
    "overlap_ratio",
    "contains_point",
    "contains_region",
    "clip_region",
]
