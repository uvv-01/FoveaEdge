"""Spatial region representation for FoveaEdge.

Provides Region, RegionSet, geometry utilities, and ROI extraction.
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
from foveaedge.regions.extraction import ExtractedROI, ROIExtractor

__all__ = [
    "Region",
    "RegionSet",
    "intersection_area",
    "iou",
    "overlap_ratio",
    "contains_point",
    "contains_region",
    "clip_region",
    "ExtractedROI",
    "ROIExtractor",
]
