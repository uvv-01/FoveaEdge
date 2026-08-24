"""RegionSet — a collection of spatial regions.

Provides safe operations for managing, iterating, and computing
aggregate properties over a set of Region objects.
"""

from __future__ import annotations

from typing import Iterator

from foveaedge.regions.region import Region


class RegionSet:
    """An ordered collection of spatial regions.

    Supports add, remove, iteration, and aggregate computations.

    Usage:
        rs = RegionSet()
        rs.add(Region(0, 0, 100, 100))
        rs.add(Region(50, 50, 100, 100))
        print(rs.total_area)
        print(rs.bounding_box)
    """

    def __init__(self, regions: list[Region] | None = None) -> None:
        """Initialize with optional list of regions.

        Args:
            regions: Initial list of regions.
        """
        self._regions: list[Region] = list(regions) if regions else []

    def add(self, region: Region) -> None:
        """Add a region to the set.

        Args:
            region: Region to add.
        """
        self._regions.append(region)

    def remove(self, index: int) -> Region:
        """Remove and return a region by index.

        Args:
            index: Index of the region to remove.

        Returns:
            The removed region.

        Raises:
            IndexError: If index is out of range.
        """
        return self._regions.pop(index)

    def clear(self) -> None:
        """Remove all regions."""
        self._regions.clear()

    def __len__(self) -> int:
        """Number of regions in the set."""
        return len(self._regions)

    def __iter__(self) -> Iterator[Region]:
        """Iterate over regions."""
        return iter(self._regions)

    def __getitem__(self, index: int) -> Region:
        """Get a region by index."""
        return self._regions[index]

    def __bool__(self) -> bool:
        """True if the set is non-empty."""
        return len(self._regions) > 0

    @property
    def total_area(self) -> int:
        """Sum of all region areas (may double-count overlaps)."""
        return sum(r.area for r in self._regions)

    @property
    def bounding_box(self) -> Region | None:
        """Compute the minimal bounding box containing all regions.

        Returns:
            Region bounding box, or None if empty.
        """
        if not self._regions:
            return None

        min_x = min(r.x for r in self._regions)
        min_y = min(r.y for r in self._regions)
        max_x2 = max(r.x2 for r in self._regions)
        max_y2 = max(r.y2 for r in self._regions)

        return Region(x=min_x, y=min_y, width=max_x2 - min_x, height=max_y2 - min_y)

    @property
    def union_area(self) -> int:
        """Approximate union area (pixel-grid based for small regions).

        For research purposes, this counts unique pixels covered
        by any region. Uses a simple bounding-box union for efficiency.
        """
        if not self._regions:
            return 0
        if len(self._regions) == 1:
            return self._regions[0].area

        # Simple approximation: use bounding box
        bbox = self.bounding_box
        return bbox.area if bbox else 0

    def filter_by_area(self, min_area: int) -> RegionSet:
        """Return a new RegionSet with only regions >= min_area.

        Args:
            min_area: Minimum area threshold.

        Returns:
            New RegionSet with filtered regions.
        """
        return RegionSet([r for r in self._regions if r.area >= min_area])

    def filter_by_bounds(
        self, frame_width: int, frame_height: int
    ) -> RegionSet:
        """Return a new RegionSet with only regions fully inside frame bounds.

        Args:
            frame_width: Frame width.
            frame_height: Frame height.

        Returns:
            New RegionSet with only valid regions.
        """
        return RegionSet(
            [
                r
                for r in self._regions
                if r.x >= 0 and r.y >= 0 and r.x2 <= frame_width and r.y2 <= frame_height
            ]
        )

    def __repr__(self) -> str:
        return f"RegionSet(count={len(self._regions)})"
