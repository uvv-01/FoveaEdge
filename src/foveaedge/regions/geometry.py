"""Geometry utilities for spatial region operations.

Provides intersection, IoU, overlap ratio, containment, and clipping
for Region objects.
"""

from __future__ import annotations

from foveaedge.regions.region import Region


def intersection_area(a: Region, b: Region) -> int:
    """Compute the area of intersection between two regions.

    Args:
        a: First region.
        b: Second region.

    Returns:
        Area of intersection in pixels. 0 if no overlap.
    """
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (ix2 - ix1) * (iy2 - iy1)


def iou(a: Region, b: Region) -> float:
    """Compute Intersection over Union (IoU) of two regions.

    Args:
        a: First region.
        b: Second region.

    Returns:
        IoU value in [0.0, 1.0]. 0.0 means no overlap, 1.0 means identical.
    """
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0

    union = a.area + b.area - inter
    if union <= 0:
        return 0.0

    return inter / union


def overlap_ratio(a: Region, b: Region) -> float:
    """Compute the ratio of intersection area to area of region a.

    Useful for measuring how much of region a is covered by region b.

    Args:
        a: Region to measure coverage of.
        b: Region to measure coverage by.

    Returns:
        Ratio in [0.0, 1.0]. 0.0 means no overlap, 1.0 means a is fully inside b.
    """
    if a.area <= 0:
        return 0.0
    inter = intersection_area(a, b)
    return inter / a.area


def contains_point(region: Region, px: int, py: int) -> bool:
    """Check if a point is inside a region.

    Args:
        region: The region to check.
        px: Point x coordinate.
        py: Point y coordinate.

    Returns:
        True if the point is within the region bounds.
    """
    return region.contains_point(px, py)


def contains_region(outer: Region, inner: Region) -> bool:
    """Check if outer fully contains inner.

    Args:
        outer: The containing region.
        inner: The region to check containment of.

    Returns:
        True if inner is fully inside outer.
    """
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x2 <= outer.x2
        and inner.y2 <= outer.y2
    )


def clip_region(region: Region, frame_width: int, frame_height: int) -> Region:
    """Clip a region to fit within frame bounds.

    Args:
        region: Region to clip.
        frame_width: Frame width.
        frame_height: Frame height.

    Returns:
        New Region clipped to frame bounds.
    """
    return region.clip_to_frame(frame_width, frame_height)


def distance_between_centers(a: Region, b: Region) -> float:
    """Compute Euclidean distance between region centers.

    Args:
        a: First region.
        b: Second region.

    Returns:
        Distance in pixels.
    """
    import math

    ax, ay = a.center
    bx, by = b.center
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
