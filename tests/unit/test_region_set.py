"""Unit tests for foveaedge.regions.region_set — RegionSet."""

from __future__ import annotations

import pytest

from foveaedge.regions.region import Region
from foveaedge.regions.region_set import RegionSet


class TestRegionSet:
    """Tests for RegionSet."""

    def test_empty(self):
        """Empty RegionSet has length 0."""
        rs = RegionSet()
        assert len(rs) == 0
        assert bool(rs) is False

    def test_add(self):
        """add increases length."""
        rs = RegionSet()
        rs.add(Region(0, 0, 10, 10))
        assert len(rs) == 1
        assert bool(rs) is True

    def test_add_multiple(self):
        """Multiple adds accumulate."""
        rs = RegionSet()
        rs.add(Region(0, 0, 10, 10))
        rs.add(Region(20, 20, 10, 10))
        assert len(rs) == 2

    def test_remove(self):
        """remove returns and removes a region."""
        rs = RegionSet()
        r = Region(0, 0, 10, 10)
        rs.add(r)
        removed = rs.remove(0)
        assert removed == r
        assert len(rs) == 0

    def test_remove_index_error(self):
        """remove with invalid index raises IndexError."""
        rs = RegionSet()
        with pytest.raises(IndexError):
            rs.remove(0)

    def test_clear(self):
        """clear empties the set."""
        rs = RegionSet([Region(0, 0, 10, 10), Region(20, 20, 10, 10)])
        rs.clear()
        assert len(rs) == 0

    def test_iter(self):
        """Iteration yields regions."""
        r1 = Region(0, 0, 10, 10)
        r2 = Region(20, 20, 10, 10)
        rs = RegionSet([r1, r2])
        regions = list(rs)
        assert regions == [r1, r2]

    def test_getitem(self):
        """Indexing returns correct region."""
        r1 = Region(0, 0, 10, 10)
        r2 = Region(20, 20, 10, 10)
        rs = RegionSet([r1, r2])
        assert rs[0] == r1
        assert rs[1] == r2

    def test_total_area(self):
        """total_area sums all regions."""
        rs = RegionSet([
            Region(0, 0, 10, 10),  # area=100
            Region(20, 20, 20, 20),  # area=400
        ])
        assert rs.total_area == 500

    def test_bounding_box_empty(self):
        """Empty set returns None for bounding_box."""
        rs = RegionSet()
        assert rs.bounding_box is None

    def test_bounding_box_single(self):
        """Single region bounding box is the region itself."""
        r = Region(10, 20, 30, 40)
        rs = RegionSet([r])
        bbox = rs.bounding_box
        assert bbox == r

    def test_bounding_box_multiple(self):
        """Bounding box encompasses all regions."""
        rs = RegionSet([
            Region(0, 0, 10, 10),
            Region(50, 50, 20, 20),
        ])
        bbox = rs.bounding_box
        assert bbox is not None
        assert bbox.x == 0
        assert bbox.y == 0
        assert bbox.x2 == 70
        assert bbox.y2 == 70

    def test_filter_by_area(self):
        """filter_by_area returns only regions meeting threshold."""
        rs = RegionSet([
            Region(0, 0, 5, 5),  # area=25
            Region(10, 10, 20, 20),  # area=400
            Region(30, 30, 10, 10),  # area=100
        ])
        filtered = rs.filter_by_area(50)
        assert len(filtered) == 2
        assert filtered[0].area >= 50
        assert filtered[1].area >= 50

    def test_filter_by_bounds(self):
        """filter_by_bounds returns only regions inside frame."""
        rs = RegionSet([
            Region(0, 0, 50, 50),  # inside 100x100
            Region(80, 80, 30, 30),  # partially outside
        ])
        filtered = rs.filter_by_bounds(100, 100)
        assert len(filtered) == 1

    def test_init_with_list(self):
        """RegionSet can be initialized with a list."""
        regions = [Region(0, 0, 10, 10), Region(20, 20, 10, 10)]
        rs = RegionSet(regions)
        assert len(rs) == 2

    def test_repr(self):
        """repr contains count."""
        rs = RegionSet([Region(0, 0, 10, 10)])
        assert "1" in repr(rs)
