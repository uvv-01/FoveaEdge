"""Unit tests for foveaedge.regions.geometry — spatial utilities."""

from __future__ import annotations

import math

import pytest

from foveaedge.regions.geometry import (
    clip_region,
    contains_point,
    contains_region,
    distance_between_centers,
    intersection_area,
    iou,
    overlap_ratio,
)
from foveaedge.regions.region import Region


class TestIntersectionArea:
    """Tests for intersection_area."""

    def test_partial_overlap(self):
        """Partial overlap computes correct area."""
        a = Region(0, 0, 100, 100)
        b = Region(50, 50, 100, 100)
        assert intersection_area(a, b) == 2500  # 50x50 overlap

    def test_no_overlap(self):
        """Non-overlapping regions return 0."""
        a = Region(0, 0, 10, 10)
        b = Region(20, 20, 10, 10)
        assert intersection_area(a, b) == 0

    def test_identical_regions(self):
        """Identical regions return the full area."""
        a = Region(10, 10, 50, 50)
        assert intersection_area(a, a) == 2500

    def test_contained(self):
        """Fully contained region returns inner area."""
        outer = Region(0, 0, 100, 100)
        inner = Region(10, 10, 20, 20)
        assert intersection_area(outer, inner) == 400

    def test_touching_edges(self):
        """Touching edges return 0 (not overlapping)."""
        a = Region(0, 0, 10, 10)
        b = Region(10, 0, 10, 10)
        assert intersection_area(a, b) == 0

    def test_touching_corners(self):
        """Touching corners return 0."""
        a = Region(0, 0, 10, 10)
        b = Region(10, 10, 10, 10)
        assert intersection_area(a, b) == 0


class TestIoU:
    """Tests for iou (Intersection over Union)."""

    def test_identical(self):
        """Identical regions have IoU = 1.0."""
        a = Region(0, 0, 100, 100)
        assert iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping regions have IoU = 0.0."""
        a = Region(0, 0, 10, 10)
        b = Region(20, 20, 10, 10)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap gives correct IoU."""
        a = Region(0, 0, 100, 100)
        b = Region(50, 0, 100, 100)
        # Intersection: 50x100 = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU: 5000/15000 = 1/3
        assert iou(a, b) == pytest.approx(1 / 3)

    def test_contained(self):
        """Fully contained gives IoU = inner/outer area ratio."""
        outer = Region(0, 0, 100, 100)
        inner = Region(0, 0, 50, 50)
        # Intersection: 2500, Union: 10000 + 2500 - 2500 = 10000
        assert iou(outer, inner) == pytest.approx(0.25)

    def test_symmetric(self):
        """IoU is symmetric."""
        a = Region(0, 0, 100, 100)
        b = Region(50, 50, 100, 100)
        assert iou(a, b) == pytest.approx(iou(b, a))


class TestOverlapRatio:
    """Tests for overlap_ratio."""

    def test_full_coverage(self):
        """Inner fully inside outer gives ratio = 1.0."""
        outer = Region(0, 0, 100, 100)
        inner = Region(10, 10, 20, 20)
        assert overlap_ratio(inner, outer) == pytest.approx(1.0)

    def test_no_coverage(self):
        """Non-overlapping gives ratio = 0.0."""
        a = Region(0, 0, 10, 10)
        b = Region(20, 20, 10, 10)
        assert overlap_ratio(a, b) == pytest.approx(0.0)

    def test_half_coverage(self):
        """Half overlap gives ratio = 0.5."""
        a = Region(0, 0, 100, 100)
        b = Region(50, 0, 100, 100)
        # Intersection: 50x100 = 5000, a.area = 10000
        assert overlap_ratio(a, b) == pytest.approx(0.5)

    def test_zero_area(self):
        """Zero area returns 0."""
        a = Region(0, 0, 1, 1)
        assert overlap_ratio(a, a) == pytest.approx(1.0)


class TestContainsPoint:
    """Tests for contains_point."""

    def test_inside(self):
        """Point inside returns True."""
        r = Region(10, 10, 50, 50)
        assert contains_point(r, 30, 30) is True

    def test_outside(self):
        """Point outside returns False."""
        r = Region(10, 10, 50, 50)
        assert contains_point(r, 0, 0) is False

    def test_boundary(self):
        """Point on boundary is handled consistently."""
        r = Region(10, 10, 50, 50)
        assert contains_point(r, 10, 10) is True  # top-left
        assert contains_point(r, 59, 59) is True  # inside bottom-right
        assert contains_point(r, 60, 10) is False  # right boundary


class TestContainsRegion:
    """Tests for contains_region."""

    def test_fully_contained(self):
        """Inner fully inside outer."""
        outer = Region(0, 0, 100, 100)
        inner = Region(10, 10, 20, 20)
        assert contains_region(outer, inner) is True

    def test_not_contained(self):
        """Inner partially outside."""
        outer = Region(0, 0, 100, 100)
        inner = Region(50, 50, 100, 100)
        assert contains_region(outer, inner) is False

    def test_identical(self):
        """Identical regions are contained."""
        r = Region(10, 10, 50, 50)
        assert contains_region(r, r) is True

    def test_boundary_contained(self):
        """Inner touching outer boundary is contained."""
        outer = Region(0, 0, 100, 100)
        inner = Region(0, 0, 100, 100)
        assert contains_region(outer, inner) is True


class TestClipRegion:
    """Tests for clip_region."""

    def test_no_clip_needed(self):
        """Region inside frame is unchanged."""
        r = Region(10, 10, 20, 20)
        clipped = clip_region(r, 100, 100)
        assert clipped == r

    def test_clip_right(self):
        """Region extending past right is clipped."""
        r = Region(80, 10, 50, 20)
        clipped = clip_region(r, 100, 100)
        assert clipped.x2 == 100

    def test_clip_bottom(self):
        """Region extending past bottom is clipped."""
        r = Region(10, 80, 20, 50)
        clipped = clip_region(r, 100, 100)
        assert clipped.y2 == 100

    def test_clip_negative(self):
        """Region with negative coords is clipped to 0."""
        r = Region(-10, -10, 30, 30)
        clipped = clip_region(r, 100, 100)
        assert clipped.x >= 0
        assert clipped.y >= 0


class TestDistanceBetweenCenters:
    """Tests for distance_between_centers."""

    def test_same_center(self):
        """Same center gives distance 0."""
        r = Region(10, 10, 20, 20)
        assert distance_between_centers(r, r) == pytest.approx(0.0)

    def test_known_distance(self):
        """Known distance is computed correctly."""
        a = Region(0, 0, 10, 10)  # center (5, 5)
        b = Region(30, 0, 10, 10)  # center (35, 5)
        assert distance_between_centers(a, b) == pytest.approx(30.0)

    def test_diagonal(self):
        """Diagonal distance uses Euclidean formula."""
        a = Region(0, 0, 10, 10)  # center (5, 5)
        b = Region(30, 40, 10, 10)  # center (35, 45)
        expected = math.sqrt(30**2 + 40**2)
        assert distance_between_centers(a, b) == pytest.approx(expected)
