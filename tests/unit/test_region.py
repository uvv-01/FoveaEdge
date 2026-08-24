"""Unit tests for foveaedge.regions.region — Region."""

from __future__ import annotations

import pytest

from foveaedge.regions.region import Region


class TestRegion:
    """Tests for Region dataclass."""

    def test_creation(self):
        """Region stores x, y, width, height."""
        r = Region(x=10, y=20, width=100, height=50)
        assert r.x == 10
        assert r.y == 20
        assert r.width == 100
        assert r.height == 50

    def test_frozen(self):
        """Region is immutable."""
        r = Region(x=0, y=0, width=10, height=10)
        with pytest.raises(AttributeError):
            r.x = 5  # type: ignore[misc]

    def test_invalid_width(self):
        """Zero width raises ValueError."""
        with pytest.raises(ValueError, match="width"):
            Region(x=0, y=0, width=0, height=10)

    def test_negative_width(self):
        """Negative width raises ValueError."""
        with pytest.raises(ValueError, match="width"):
            Region(x=0, y=0, width=-5, height=10)

    def test_invalid_height(self):
        """Zero height raises ValueError."""
        with pytest.raises(ValueError, match="height"):
            Region(x=0, y=0, width=10, height=0)

    def test_negative_height(self):
        """Negative height raises ValueError."""
        with pytest.raises(ValueError, match="height"):
            Region(x=0, y=0, width=10, height=-5)

    def test_x2(self):
        """x2 returns x + width."""
        r = Region(x=10, y=20, width=100, height=50)
        assert r.x2 == 110

    def test_y2(self):
        """y2 returns y + height."""
        r = Region(x=10, y=20, width=100, height=50)
        assert r.y2 == 70

    def test_area(self):
        """area returns width * height."""
        r = Region(x=0, y=0, width=10, height=20)
        assert r.area == 200

    def test_center(self):
        """center returns (cx, cy)."""
        r = Region(x=10, y=20, width=100, height=50)
        cx, cy = r.center
        assert cx == 60
        assert cy == 45

    def test_xyxy(self):
        """xyxy returns (x1, y1, x2, y2)."""
        r = Region(x=10, y=20, width=100, height=50)
        assert r.xyxy == (10, 20, 110, 70)

    def test_from_xyxy(self):
        """from_xyxy creates correct region."""
        r = Region.from_xyxy(10, 20, 110, 70)
        assert r.x == 10
        assert r.y == 20
        assert r.width == 100
        assert r.height == 50

    def test_from_xyxy_invalid(self):
        """from_xyxy with x2 <= x1 raises ValueError."""
        with pytest.raises(ValueError, match="x2"):
            Region.from_xyxy(10, 20, 5, 70)

    def test_from_xyxy_invalid_y(self):
        """from_xyxy with y2 <= y1 raises ValueError."""
        with pytest.raises(ValueError, match="y2"):
            Region.from_xyxy(10, 20, 110, 10)

    def test_from_center(self):
        """from_center creates centered region."""
        r = Region.from_center(cx=50, cy=50, width=20, height=20)
        assert r.x == 40
        assert r.y == 40
        assert r.width == 20
        assert r.height == 20
        assert r.center == (50, 50)

    def test_from_center_odd(self):
        """from_center with odd dimensions."""
        r = Region.from_center(cx=50, cy=50, width=21, height=21)
        assert r.center == (50, 50)

    def test_contains_point_inside(self):
        """contains_point returns True for interior points."""
        r = Region(x=10, y=10, width=100, height=100)
        assert r.contains_point(50, 50) is True

    def test_contains_point_boundary(self):
        """contains_point is True on left/top boundary, False on right/bottom."""
        r = Region(x=10, y=10, width=100, height=100)
        assert r.contains_point(10, 10) is True  # top-left
        assert r.contains_point(109, 109) is True  # inside bottom-right
        assert r.contains_point(110, 10) is False  # right boundary (exclusive)
        assert r.contains_point(10, 110) is False  # bottom boundary (exclusive)

    def test_contains_point_outside(self):
        """contains_point returns False for exterior points."""
        r = Region(x=10, y=10, width=100, height=100)
        assert r.contains_point(0, 0) is False
        assert r.contains_point(200, 200) is False

    def test_intersects(self):
        """intersects detects overlapping regions."""
        a = Region(x=0, y=0, width=100, height=100)
        b = Region(x=50, y=50, width=100, height=100)
        assert a.intersects(b) is True
        assert b.intersects(a) is True

    def test_no_intersection(self):
        """intersects returns False for non-overlapping regions."""
        a = Region(x=0, y=0, width=10, height=10)
        b = Region(x=20, y=20, width=10, height=10)
        assert a.intersects(b) is False

    def test_touching_edges_no_intersection(self):
        """Touching edges (not overlapping) return False."""
        a = Region(x=0, y=0, width=10, height=10)
        b = Region(x=10, y=0, width=10, height=10)
        assert a.intersects(b) is False

    def test_identical_regions_intersect(self):
        """Identical regions intersect."""
        a = Region(x=5, y=5, width=10, height=10)
        assert a.intersects(a) is True

    def test_clip_to_frame(self):
        """clip_to_frame restricts region to frame bounds."""
        r = Region(x=50, y=50, width=200, height=200)
        clipped = r.clip_to_frame(100, 100)
        assert clipped.x == 50
        assert clipped.y == 50
        assert clipped.x2 == 100
        assert clipped.y2 == 100

    def test_clip_to_frame_negative(self):
        """clip_to_frame handles negative coordinates."""
        r = Region(x=-10, y=-10, width=20, height=20)
        clipped = r.clip_to_frame(100, 100)
        assert clipped.x >= 0
        assert clipped.y >= 0
        assert clipped.x2 <= 100
        assert clipped.y2 <= 100

    def test_clip_to_frame_fully_outside(self):
        """clip_to_frame returns minimal region for fully outside."""
        r = Region(x=200, y=200, width=10, height=10)
        clipped = r.clip_to_frame(100, 100)
        assert clipped.width >= 1
        assert clipped.height >= 1

    def test_clip_invalid_frame(self):
        """clip_to_frame raises ValueError for invalid frame."""
        r = Region(x=0, y=0, width=10, height=10)
        with pytest.raises(ValueError):
            r.clip_to_frame(0, 100)

    def test_repr(self):
        """repr contains key attributes."""
        r = Region(x=1, y=2, width=3, height=4)
        s = repr(r)
        assert "1" in s
        assert "2" in s
        assert "3" in s
        assert "4" in s
