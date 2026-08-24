"""Unit tests for foveaedge.regions.extraction — ROIExtractor and ExtractedROI."""

from __future__ import annotations

import numpy as np
import pytest

from foveaedge.regions.extraction import ExtractedROI, ROIExtractor
from foveaedge.regions.region import Region


# ---------------------------------------------------------------------------
# Helper: deterministic test image
# ---------------------------------------------------------------------------

def _make_test_image(height: int = 100, width: int = 100, channels: int = 3, dtype=np.uint8) -> np.ndarray:
    """Create a deterministic test image where each pixel's value encodes its position.

    For a 3-channel image, pixel (y, x) gets values [y % 256, x % 256, (y+x) % 256].
    For grayscale, pixel (y, x) gets (y * width + x) % 256.
    """
    if channels == 1:
        img = np.zeros((height, width), dtype=dtype)
        for y in range(height):
            for x in range(width):
                img[y, x] = (y * width + x) % 256
        return img
    else:
        img = np.zeros((height, width, channels), dtype=dtype)
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = y % 256
                img[y, x, 1] = x % 256
                img[y, x, 2] = (y + x) % 256
        return img


# ---------------------------------------------------------------------------
# ExtractedROI tests
# ---------------------------------------------------------------------------


class TestExtractedROI:
    """Tests for ExtractedROI."""

    def test_creation(self):
        """ExtractedROI stores image, region, and optional index."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        region = Region(0, 0, 10, 10)
        roi = ExtractedROI(image=img, region=region, index=0)
        assert roi.image is img
        assert roi.region == region
        assert roi.index == 0

    def test_shape(self):
        """shape returns the image shape."""
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        roi = ExtractedROI(image=img, region=Region(0, 0, 30, 20))
        assert roi.shape == (20, 30, 3)

    def test_dtype(self):
        """dtype returns the image dtype."""
        img = np.zeros((10, 10), dtype=np.float32)
        roi = ExtractedROI(image=img, region=Region(0, 0, 10, 10))
        assert roi.dtype == np.float32

    def test_area(self):
        """area returns the region area."""
        roi = ExtractedROI(image=np.zeros((10, 10)), region=Region(0, 0, 10, 10))
        assert roi.area == 100

    def test_index_optional(self):
        """index defaults to None."""
        roi = ExtractedROI(image=np.zeros((5, 5)), region=Region(0, 0, 5, 5))
        assert roi.index is None

    def test_repr(self):
        """repr contains region, shape, dtype."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        roi = ExtractedROI(image=img, region=Region(5, 5, 10, 10), index=2)
        r = repr(roi)
        assert "ExtractedROI" in r
        assert "10" in r
        assert "2" in r


# ---------------------------------------------------------------------------
# ROIExtractor tests
# ---------------------------------------------------------------------------


class TestROIExtractor:
    """Tests for ROIExtractor."""

    @pytest.fixture
    def extractor(self) -> ROIExtractor:
        return ROIExtractor()

    @pytest.fixture
    def image(self) -> np.ndarray:
        return _make_test_image(100, 100, 3)

    # --- Basic extraction ---

    def test_top_left_roi(self, extractor: ROIExtractor, image: np.ndarray):
        """Extract from top-left corner."""
        region = Region(0, 0, 10, 10)
        roi = extractor.extract(image, region)
        assert roi.shape == (10, 10, 3)
        assert roi.region == region

    def test_center_roi(self, extractor: ROIExtractor, image: np.ndarray):
        """Extract from center."""
        region = Region(40, 40, 20, 20)
        roi = extractor.extract(image, region)
        assert roi.shape == (20, 20, 3)
        assert roi.region == region

    def test_bottom_right_roi(self, extractor: ROIExtractor, image: np.ndarray):
        """Extract from bottom-right."""
        region = Region(80, 80, 20, 20)
        roi = extractor.extract(image, region)
        assert roi.shape == (20, 20, 3)
        assert roi.region == region

    def test_single_pixel_roi(self, extractor: ROIExtractor, image: np.ndarray):
        """Extract a single pixel."""
        region = Region(5, 5, 1, 1)
        roi = extractor.extract(image, region)
        assert roi.shape == (1, 1, 3)

    def test_full_frame_roi(self, extractor: ROIExtractor, image: np.ndarray):
        """Extract the entire frame."""
        region = Region(0, 0, 100, 100)
        roi = extractor.extract(image, region)
        assert roi.shape == (100, 100, 3)
        np.testing.assert_array_equal(roi.image, image)

    # --- Content correctness ---

    def test_content_correctness(self, extractor: ROIExtractor):
        """Extracted crop matches expected pixel values."""
        img = _make_test_image(10, 10, 3)
        region = Region(2, 3, 4, 5)
        roi = extractor.extract(img, region)

        # Verify each pixel in the crop
        for dy in range(5):
            for dx in range(4):
                py = 3 + dy
                px = 2 + dx
                np.testing.assert_array_equal(
                    roi.image[dy, dx],
                    img[py, px],
                    err_msg=f"Pixel mismatch at ({dy}, {dx}) -> frame ({py}, {px})",
                )

    def test_content_top_left(self, extractor: ROIExtractor):
        """Top-left crop has correct values."""
        img = _make_test_image(20, 20, 3)
        roi = extractor.extract(img, Region(0, 0, 5, 5))
        # Pixel (0,0) should be [0, 0, 0]
        np.testing.assert_array_equal(roi.image[0, 0], [0, 0, 0])
        # Pixel (1,2) should be [1, 2, 3]
        np.testing.assert_array_equal(roi.image[1, 2], [1, 2, 3])

    # --- Shape verification ---

    def test_roi_height_matches_region(self, extractor: ROIExtractor, image: np.ndarray):
        """ROI height equals region height for valid regions."""
        for h in [1, 5, 10, 50, 100]:
            roi = extractor.extract(image, Region(0, 0, 10, h))
            assert roi.shape[0] == h

    def test_roi_width_matches_region(self, extractor: ROIExtractor, image: np.ndarray):
        """ROI width equals region width for valid regions."""
        for w in [1, 5, 10, 50, 100]:
            roi = extractor.extract(image, Region(0, 0, w, 10))
            assert roi.shape[1] == w

    # --- dtype preservation ---

    def test_preserves_uint8(self, extractor: ROIExtractor):
        """uint8 frame produces uint8 ROI."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        roi = extractor.extract(img, Region(0, 0, 5, 5))
        assert roi.dtype == np.uint8

    def test_preserves_float32(self, extractor: ROIExtractor):
        """float32 frame produces float32 ROI."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        roi = extractor.extract(img, Region(0, 0, 5, 5))
        assert roi.dtype == np.float32

    def test_preserves_grayscale(self, extractor: ROIExtractor):
        """Grayscale frame produces 2D ROI."""
        img = np.zeros((10, 10), dtype=np.uint8)
        roi = extractor.extract(img, Region(0, 0, 5, 5))
        assert roi.shape == (5, 5)

    def test_preserves_channels(self, extractor: ROIExtractor):
        """Channel count is preserved."""
        for c in [1, 3]:
            img = np.zeros((10, 10, c), dtype=np.uint8) if c > 1 else np.zeros((10, 10), dtype=np.uint8)
            roi = extractor.extract(img, Region(0, 0, 5, 5))
            if c > 1:
                assert roi.shape[2] == c

    # --- Source immutability ---

    def test_source_frame_not_modified(self, extractor: ROIExtractor):
        """Extraction does not modify the original frame."""
        img = _make_test_image(20, 20, 3)
        original = img.copy()
        extractor.extract(img, Region(5, 5, 10, 10))
        np.testing.assert_array_equal(img, original)

    def test_source_frame_not_modified_extract_many(self, extractor: ROIExtractor):
        """extract_many does not modify the original frame."""
        img = _make_test_image(20, 20, 3)
        original = img.copy()
        extractor.extract_many(img, [Region(0, 0, 5, 5), Region(10, 10, 5, 5)])
        np.testing.assert_array_equal(img, original)

    def test_extract_returns_copy(self, extractor: ROIExtractor):
        """Extracted image is a copy, not a view."""
        img = _make_test_image(20, 20, 3)
        roi = extractor.extract(img, Region(0, 0, 10, 10))
        roi.image[:] = 0
        # Original should be unchanged
        assert img[0, 0, 0] != 0 or img[5, 5, 1] != 0

    # --- Boundary handling ---

    def test_region_touching_left_boundary(self, extractor: ROIExtractor, image: np.ndarray):
        """Region touching left boundary."""
        roi = extractor.extract(image, Region(0, 10, 10, 10))
        assert roi.shape == (10, 10, 3)
        assert roi.region.x == 0

    def test_region_touching_right_boundary(self, extractor: ROIExtractor, image: np.ndarray):
        """Region touching right boundary."""
        roi = extractor.extract(image, Region(90, 10, 10, 10))
        assert roi.shape == (10, 10, 3)
        assert roi.region.x2 == 100

    def test_region_touching_top_boundary(self, extractor: ROIExtractor, image: np.ndarray):
        """Region touching top boundary."""
        roi = extractor.extract(image, Region(10, 0, 10, 10))
        assert roi.shape == (10, 10, 3)
        assert roi.region.y == 0

    def test_region_touching_bottom_boundary(self, extractor: ROIExtractor, image: np.ndarray):
        """Region touching bottom boundary."""
        roi = extractor.extract(image, Region(10, 90, 10, 10))
        assert roi.shape == (10, 10, 3)
        assert roi.region.y2 == 100

    def test_clipping_right(self, extractor: ROIExtractor, image: np.ndarray):
        """Region extending past right boundary is clipped."""
        region = Region(85, 10, 30, 10)  # x2=115, past 100
        roi = extractor.extract(image, region)
        assert roi.region.x2 == 100
        assert roi.shape == (10, 15, 3)  # clipped width

    def test_clipping_bottom(self, extractor: ROIExtractor, image: np.ndarray):
        """Region extending past bottom boundary is clipped."""
        region = Region(10, 85, 10, 30)  # y2=115, past 100
        roi = extractor.extract(image, region)
        assert roi.region.y2 == 100
        assert roi.shape == (15, 10, 3)  # clipped height

    def test_clipping_both(self, extractor: ROIExtractor, image: np.ndarray):
        """Region extending past both boundaries is clipped."""
        region = Region(85, 85, 30, 30)
        roi = extractor.extract(image, region)
        assert roi.region.x2 == 100
        assert roi.region.y2 == 100
        assert roi.shape == (15, 15, 3)

    def test_clipping_left(self, extractor: ROIExtractor, image: np.ndarray):
        """Region with negative x is clipped."""
        region = Region(-5, 10, 20, 10)
        roi = extractor.extract(image, region)
        assert roi.region.x == 0
        assert roi.region.width == 15

    def test_clipping_top(self, extractor: ROIExtractor, image: np.ndarray):
        """Region with negative y is clipped."""
        region = Region(10, -5, 10, 20)
        roi = extractor.extract(image, region)
        assert roi.region.y == 0
        assert roi.region.height == 15

    def test_clipping_preserves_content(self, extractor: ROIExtractor):
        """Clipped region still extracts correct content."""
        img = _make_test_image(10, 10, 3)
        # Request region that extends past bottom-right
        roi = extractor.extract(img, Region(5, 5, 10, 10))
        # Should be clipped to 5x5
        assert roi.shape == (5, 5, 3)
        # Content should match original image
        np.testing.assert_array_equal(roi.image, img[5:10, 5:10])

    # --- Multiple ROIs ---

    def test_extract_many_basic(self, extractor: ROIExtractor, image: np.ndarray):
        """extract_many returns one result per region."""
        regions = [Region(0, 0, 10, 10), Region(20, 20, 10, 10)]
        rois = extractor.extract_many(image, regions)
        assert len(rois) == 2
        assert rois[0].shape == (10, 10, 3)
        assert rois[1].shape == (10, 10, 3)

    def test_extract_many_preserves_order(self, extractor: ROIExtractor, image: np.ndarray):
        """extract_many preserves input ordering."""
        regions = [
            Region(0, 0, 5, 5),
            Region(50, 50, 10, 10),
            Region(10, 10, 3, 3),
        ]
        rois = extractor.extract_many(image, regions)
        assert rois[0].region == regions[0]
        assert rois[1].region == regions[1]
        assert rois[2].region == regions[2]

    def test_extract_many_identical_regions(self, extractor: ROIExtractor, image: np.ndarray):
        """Same region supplied twice produces identical results."""
        r = Region(10, 10, 5, 5)
        rois = extractor.extract_many(image, [r, r])
        assert len(rois) == 2
        np.testing.assert_array_equal(rois[0].image, rois[1].image)

    def test_extract_many_overlapping(self, extractor: ROIExtractor, image: np.ndarray):
        """Overlapping regions are extracted independently."""
        r1 = Region(0, 0, 20, 20)
        r2 = Region(10, 10, 20, 20)
        rois = extractor.extract_many(image, [r1, r2])
        assert len(rois) == 2
        # Overlapping area should have same content
        np.testing.assert_array_equal(rois[0].image[10:20, 10:20], rois[1].image[0:10, 0:10])

    def test_extract_many_empty_list(self, extractor: ROIExtractor, image: np.ndarray):
        """Empty region list returns empty result."""
        rois = extractor.extract_many(image, [])
        assert len(rois) == 0

    # --- Error handling ---

    def test_none_frame(self, extractor: ROIExtractor):
        """None frame raises ValueError."""
        with pytest.raises(ValueError, match="must not be None"):
            extractor.extract(None, Region(0, 0, 10, 10))

    def test_none_frame_extract_many(self, extractor: ROIExtractor):
        """None frame in extract_many raises ValueError."""
        with pytest.raises(ValueError, match="must not be None"):
            extractor.extract_many(None, [Region(0, 0, 10, 10)])

    def test_invalid_dimensions_1d(self, extractor: ROIExtractor):
        """1D frame raises ValueError."""
        with pytest.raises(ValueError, match="2D or 3D"):
            extractor.extract(np.zeros(10), Region(0, 0, 5, 5))

    def test_invalid_dimensions_4d(self, extractor: ROIExtractor):
        """4D frame raises ValueError."""
        with pytest.raises(ValueError, match="2D or 3D"):
            extractor.extract(np.zeros((2, 3, 4, 5)), Region(0, 0, 2, 2))

    def test_zero_size_frame(self, extractor: ROIExtractor):
        """Zero-size frame raises ValueError."""
        img = np.zeros((0, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="invalid dimensions"):
            extractor.extract(img, Region(0, 0, 5, 5))


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestROIExtractionPipeline:
    """Integration test: Frame + Region -> ROIExtractor -> ExtractedROI."""

    def test_full_pipeline(self):
        """Complete extraction pipeline produces correct output."""
        # Create a known 50x50 RGB image
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        # Fill a 10x10 block at (10, 10) with white
        img[10:20, 10:20] = 255

        # Extract the white block
        extractor = ROIExtractor()
        region = Region(10, 10, 10, 10)
        roi = extractor.extract(img, region)

        # Verify crop shape
        assert roi.shape == (10, 10, 3)

        # Verify crop content is all white
        assert np.all(roi.image == 255)

        # Verify original coordinates preserved
        assert roi.region.x == 10
        assert roi.region.y == 10
        assert roi.region.width == 10
        assert roi.region.height == 10

        # Verify original image unchanged
        assert np.all(img[10:20, 10:20] == 255)

    def test_pipeline_with_clipping(self):
        """Pipeline clips region and preserves correct coordinates."""
        img = np.ones((30, 30, 3), dtype=np.uint8) * 128
        extractor = ROIExtractor()
        region = Region(20, 20, 20, 20)  # extends past 30x30

        roi = extractor.extract(img, region)

        # Should be clipped to 10x10
        assert roi.shape == (10, 10, 3)
        assert roi.region.x == 20
        assert roi.region.y == 20
        assert roi.region.x2 == 30
        assert roi.region.y2 == 30

    def test_pipeline_deterministic(self):
        """Same inputs produce same outputs."""
        img = _make_test_image(50, 50, 3)
        extractor = ROIExtractor()
        region = Region(10, 10, 15, 15)

        roi1 = extractor.extract(img, region)
        roi2 = extractor.extract(img, region)

        np.testing.assert_array_equal(roi1.image, roi2.image)
        assert roi1.region == roi2.region

    def test_pipeline_no_openvino(self):
        """Pipeline works without OpenVINO — pure NumPy."""
        img = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
        extractor = ROIExtractor()
        roi = extractor.extract(img, Region(5, 5, 20, 20))
        assert roi.shape == (20, 20, 3)
        # Verify it's the right content
        np.testing.assert_array_equal(roi.image, img[5:25, 5:25])
