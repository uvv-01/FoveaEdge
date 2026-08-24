"""Spatial region representation.

Provides Region dataclass for identifying rectangular areas in images/frames.

Coordinate convention:
    - Origin: top-left corner of the image
    - x increases rightward
    - y increases downward
    - Width extends rightward from x
    - Height extends downward from y

All coordinates are in pixel space (integer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class Region:
    """A rectangular region in pixel coordinates.

    Attributes:
        x: Left edge (pixels from left).
        y: Top edge (pixels from top).
        width: Width in pixels.
        height: Height in pixels.

    Convention:
        Origin is top-left. x increases rightward, y increases downward.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate region dimensions."""
        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> Self:
        """Create a Region from (x1, y1, x2, y2) format.

        Args:
            x1: Left edge.
            y1: Top edge.
            x2: Right edge (exclusive).
            y2: Bottom edge (exclusive).

        Returns:
            Region instance.

        Raises:
            ValueError: If x2 <= x1 or y2 <= y1.
        """
        if x2 <= x1:
            raise ValueError(f"x2 ({x2}) must be > x1 ({x1})")
        if y2 <= y1:
            raise ValueError(f"y2 ({y2}) must be > y1 ({y1})")
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @classmethod
    def from_center(cls, cx: int, cy: int, width: int, height: int) -> Self:
        """Create a Region centered at (cx, cy).

        Args:
            cx: Center x coordinate.
            cy: Center y coordinate.
            width: Width in pixels.
            height: Height in pixels.

        Returns:
            Region instance.
        """
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        return cls(x=cx - width // 2, y=cy - height // 2, width=width, height=height)

    @property
    def x2(self) -> int:
        """Right edge (exclusive): x + width."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge (exclusive): y + height."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point (cx, cy)."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x2, self.y2)

    def contains_point(self, px: int, py: int) -> bool:
        """Check if a point is inside this region.

        Args:
            px: Point x coordinate.
            py: Point y coordinate.

        Returns:
            True if the point is within the region bounds.
        """
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def intersects(self, other: Region) -> bool:
        """Check if this region intersects with another.

        Args:
            other: Another Region.

        Returns:
            True if the regions overlap.
        """
        return (
            self.x < other.x2
            and self.x2 > other.x
            and self.y < other.y2
            and self.y2 > other.y
        )

    def clip_to_frame(self, frame_width: int, frame_height: int) -> Region:
        """Clip this region to fit within frame bounds.

        Args:
            frame_width: Width of the frame.
            frame_height: Height of the frame.

        Returns:
            New Region clipped to frame bounds.

        Raises:
            ValueError: If frame dimensions are invalid.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Frame dimensions must be > 0, got {frame_width}x{frame_height}")

        new_x = max(0, min(self.x, frame_width))
        new_y = max(0, min(self.y, frame_height))
        new_x2 = max(0, min(self.x2, frame_width))
        new_y2 = max(0, min(self.y2, frame_height))

        new_width = new_x2 - new_x
        new_height = new_y2 - new_y

        if new_width <= 0 or new_height <= 0:
            # Region is entirely outside frame
            return Region(x=0, y=0, width=1, height=1)

        return Region(x=new_x, y=new_y, width=new_width, height=new_height)

    def __repr__(self) -> str:
        return f"Region(x={self.x}, y={self.y}, w={self.width}, h={self.height})"
