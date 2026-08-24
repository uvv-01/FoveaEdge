"""ROI extraction from image frames.

Provides ExtractedROI for representing extracted crops with their
original coordinate metadata, and ROIExtractor for performing the
actual extraction.

Coordinate convention (inherited from Region):
    - Origin: top-left corner of the image
    - x increases rightward
    - y increases downward
    - Half-open bounds: [x, x+width), [y, y+height)

Boundary behavior:
    - Regions fully inside the frame are extracted directly.
    - Regions partially outside the frame are clipped to frame bounds
      using Region.clip_to_frame() before extraction.
    - The ExtractedROI.region reflects the *clipped* region actually used,
      not the original unclipped region. This ensures coordinate accuracy
      for later result mapping.

Source safety:
    - ROIExtractor never mutates the source frame.

This module is independent of OpenVINO, detectors, trackers, and selectors.
"""

from __future__ import annotations

import numpy as np

from foveaedge.regions.region import Region


class ExtractedROI:
    """An extracted image crop with its original frame coordinates.

    Attributes:
        image: The extracted crop as a numpy array. Preserves the source
               frame's dtype and channel count.
        region: The Region (after clipping) that identifies where this
                crop came from in the original frame.
        index: Optional index identifier for ordering/grouping.
    """

    __slots__ = ("image", "region", "index")

    def __init__(self, image: np.ndarray, region: Region, index: int | None = None) -> None:
        """Initialize an extracted ROI.

        Args:
            image: The extracted crop numpy array.
            region: The Region identifying where this crop came from.
            index: Optional index identifier.
        """
        self.image = image
        self.region = region
        self.index = index

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the extracted image."""
        return self.image.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the extracted image."""
        return self.image.dtype

    @property
    def area(self) -> int:
        """Area of the region in pixels."""
        return self.region.area

    def __repr__(self) -> str:
        idx_str = f", index={self.index}" if self.index is not None else ""
        return (
            f"ExtractedROI(region={self.region}, "
            f"shape={self.shape}, dtype={self.dtype}{idx_str})"
        )


class ROIExtractor:
    """Extract image crops corresponding to Region objects.

    Usage:
        extractor = ROIExtractor()
        roi = extractor.extract(frame, region)
        print(roi.image.shape, roi.region)

        rois = extractor.extract_many(frame, [region1, region2])
    """

    def extract(self, frame: np.ndarray, region: Region) -> ExtractedROI:
        """Extract a single ROI from a frame.

        If the region extends outside the frame, it is clipped to frame
        bounds using Region.clip_to_frame(). The ExtractedROI.region
        reflects the clipped region.

        Args:
            frame: Source image as a numpy array (H, W) or (H, W, C).
            region: Region to extract.

        Returns:
            ExtractedROI with the cropped image and coordinate metadata.

        Raises:
            ValueError: If frame is None, has invalid dimensions, or region
                       is entirely outside the frame after clipping.
        """
        if frame is None:
            raise ValueError("frame must not be None")

        if frame.ndim < 2 or frame.ndim > 3:
            raise ValueError(f"frame must be 2D or 3D, got {frame.ndim}D")

        frame_height, frame_width = frame.shape[:2]

        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(f"frame has invalid dimensions: {frame_width}x{frame_height}")

        # Clip region to frame bounds
        clipped = region.clip_to_frame(frame_width, frame_height)

        # Extract the crop using half-open bounds
        # Region convention: x is column offset, y is row offset
        crop = frame[clipped.y : clipped.y2, clipped.x : clipped.x2].copy()

        return ExtractedROI(image=crop, region=clipped)

    def extract_many(
        self, frame: np.ndarray, regions: list[Region]
    ) -> list[ExtractedROI]:
        """Extract multiple ROIs from a frame.

        Preserves input ordering. Each region is independently clipped
        and extracted.

        Args:
            frame: Source image as a numpy array.
            regions: List of Region objects to extract.

        Returns:
            List of ExtractedROI objects, one per input region.

        Raises:
            ValueError: If frame is None or has invalid dimensions.
        """
        if frame is None:
            raise ValueError("frame must not be None")

        return [self.extract(frame, r) for r in regions]
