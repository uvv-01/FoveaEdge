"""Unit tests for ROIInferenceEngine and ROIInferenceResult.

Tests the ROI inference pipeline: ExtractedROI -> InferenceEngine -> ROIInferenceResult.
Verifies coordinate preservation, timing, ordering, and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from foveaedge.inference.engine import InferenceEngine, InferenceResult
from foveaedge.inference.roi_engine import ROIInferenceEngine, ROIInferenceResult
from foveaedge.regions.extraction import ExtractedROI, ROIExtractor
from foveaedge.regions.region import Region

TEST_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "test_model"
TEST_MODEL_XML = TEST_MODEL_DIR / "test_model.xml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loaded_engine() -> InferenceEngine:
    """Load test model and compile on CPU."""
    if not TEST_MODEL_XML.exists():
        pytest.skip("Test model XML not generated")
    from foveaedge.model import ModelLoader

    loader = ModelLoader()
    loader.load_and_compile(str(TEST_MODEL_XML), device="CPU")
    return InferenceEngine(loader)


@pytest.fixture
def roi_engine(loaded_engine: InferenceEngine) -> ROIInferenceEngine:
    """Create ROIInferenceEngine from loaded engine."""
    return ROIInferenceEngine(loaded_engine)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Deterministic 64x64 RGB frame for testing."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def extractor() -> ROIExtractor:
    """Create a standard ROIExtractor."""
    return ROIExtractor()


# ---------------------------------------------------------------------------
# ROIInferenceResult tests
# ---------------------------------------------------------------------------

class TestROIInferenceResult:
    """Tests for ROIInferenceResult dataclass."""

    def _make_result(self, region: Region, roi_index: int | None = None) -> ROIInferenceResult:
        """Helper to create an ROIInferenceResult with mock inference."""
        inference_result = InferenceResult(
            output_tensors={"output": np.array([[0.1, 0.9]])},
            preprocessing_time_s=0.001,
            inference_time_s=0.005,
            postprocessing_time_s=0.0005,
            total_time_s=0.0065,
            input_shape=(1, 3, 32, 32),
            model_name="test_model",
            device="CPU",
        )
        return ROIInferenceResult(
            inference_result=inference_result,
            region=region,
            extraction_time_s=0.002,
            roi_index=roi_index,
        )

    def test_basic_properties(self):
        """Properties delegate to underlying InferenceResult."""
        region = Region(x=10, y=20, width=32, height=32)
        result = self._make_result(region)
        assert result.region == region
        np.testing.assert_array_equal(result.output_tensors["output"], np.array([[0.1, 0.9]]))
        assert result.first_output is not None
        assert result.model_name == "test_model"
        assert result.device == "CPU"

    def test_timing_fields(self):
        """Timing fields are correctly reported."""
        region = Region(x=0, y=0, width=32, height=32)
        result = self._make_result(region)
        assert result.preprocessing_time_s == pytest.approx(0.001)
        assert result.inference_time_s == pytest.approx(0.005)
        assert result.postprocessing_time_s == pytest.approx(0.0005)
        assert result.total_inference_time_s == pytest.approx(0.0065)
        assert result.extraction_time_s == pytest.approx(0.002)

    def test_total_time_includes_extraction(self):
        """total_time_s = extraction_time + inference total."""
        region = Region(x=0, y=0, width=32, height=32)
        result = self._make_result(region)
        expected = result.extraction_time_s + result.total_inference_time_s
        assert result.total_time_s == pytest.approx(expected)

    def test_roi_index_preserved(self):
        """roi_index is preserved when set."""
        region = Region(x=0, y=0, width=32, height=32)
        result = self._make_result(region, roi_index=3)
        assert result.roi_index == 3

    def test_roi_index_none(self):
        """roi_index defaults to None."""
        region = Region(x=0, y=0, width=32, height=32)
        result = self._make_result(region)
        assert result.roi_index is None

    def test_repr(self):
        """repr includes region and timing."""
        region = Region(x=10, y=20, width=32, height=32)
        result = self._make_result(region, roi_index=1)
        r = repr(result)
        assert "ROIInferenceResult" in r
        assert "roi_index=1" in r

    def test_region_not_mutated(self):
        """Region reference is preserved, not replaced."""
        region = Region(x=100, y=200, width=32, height=32)
        result = self._make_result(region)
        assert result.region.x == 100
        assert result.region.y == 200


# ---------------------------------------------------------------------------
# ROIInferenceEngine — single ROI tests
# ---------------------------------------------------------------------------

class TestROIInferenceEngineSingleROI:
    """Tests for single ROI inference."""

    def test_infer_roi_basic(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Basic single ROI inference produces valid result."""
        region = Region(x=0, y=0, width=32, height=32)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        assert result.first_output is not None
        assert result.first_output.shape == (1, 10)
        assert result.region == region
        assert result.model_name == "test_model"

    def test_infer_roi_preserves_original_coordinates(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Region coordinates are preserved through inference."""
        region = Region(x=16, y=24, width=32, height=32)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        assert result.region.x == 16
        assert result.region.y == 24
        assert result.region.width == 32
        assert result.region.height == 32

    def test_infer_roi_center_of_frame(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """ROI from center of frame works correctly."""
        region = Region(x=16, y=16, width=32, height=32)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        assert result.first_output is not None
        assert result.region == region

    def test_infer_roi_top_left(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """ROI at top-left corner."""
        region = Region(x=0, y=0, width=16, height=16)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        assert result.first_output is not None
        assert result.region == region

    def test_infer_roi_bottom_right(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """ROI at bottom-right corner."""
        region = Region(x=48, y=48, width=16, height=16)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        assert result.first_output is not None
        assert result.region == region

    def test_infer_roi_timing_non_negative(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """All timing fields are non-negative."""
        region = Region(x=0, y=0, width=32, height=32)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        assert result.preprocessing_time_s >= 0
        assert result.inference_time_s > 0
        assert result.postprocessing_time_s >= 0
        assert result.total_time_s > 0

    def test_infer_roi_deterministic(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Same ROI on same frame produces same inference output."""
        region = Region(x=8, y=8, width=32, height=32)
        extracted1 = extractor.extract(sample_frame, region)
        extracted2 = extractor.extract(sample_frame, region)

        result1 = roi_engine.infer_roi(extracted1)
        result2 = roi_engine.infer_roi(extracted2)

        np.testing.assert_array_equal(result1.first_output, result2.first_output)

    def test_infer_roi_different_regions_different_outputs(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Different spatial regions produce different crops and potentially different outputs."""
        region_a = Region(x=0, y=0, width=32, height=32)
        region_b = Region(x=32, y=32, width=32, height=32)

        extracted_a = extractor.extract(sample_frame, region_a)
        extracted_b = extractor.extract(sample_frame, region_b)

        # Verify the crops are different
        assert not np.array_equal(extracted_a.image, extracted_b.image)

        result_a = roi_engine.infer_roi(extracted_a)
        result_b = roi_engine.infer_roi(extracted_b)

        # Both should produce valid outputs
        assert result_a.first_output is not None
        assert result_b.first_output is not None


# ---------------------------------------------------------------------------
# ROIInferenceEngine — error handling tests
# ---------------------------------------------------------------------------

class TestROIInferenceEngineErrors:
    """Tests for error handling in ROIInferenceEngine."""

    def test_infer_roi_none_raises(self, roi_engine: ROIInferenceEngine):
        """infer_roi raises ValueError for None input."""
        with pytest.raises(ValueError, match="must not be None"):
            roi_engine.infer_roi(None)  # type: ignore[arg-type]

    def test_infer_roi_empty_image_raises(self, roi_engine: ROIInferenceEngine):
        """infer_roi raises ValueError for empty image."""
        region = Region(x=0, y=0, width=10, height=10)
        empty_roi = ExtractedROI(image=np.array([]), region=region)
        with pytest.raises(ValueError, match="empty or None"):
            roi_engine.infer_roi(empty_roi)

    def test_infer_rois_empty_list_raises(self, roi_engine: ROIInferenceEngine):
        """infer_rois raises ValueError for empty list."""
        with pytest.raises(ValueError, match="must not be empty"):
            roi_engine.infer_rois([])

    def test_infer_from_regions_none_frame_raises(self, roi_engine: ROIInferenceEngine):
        """infer_from_regions raises ValueError for None frame."""
        region = Region(x=0, y=0, width=32, height=32)
        with pytest.raises(ValueError, match="frame must not be None"):
            roi_engine.infer_from_regions(None, [region])  # type: ignore[arg-type]

    def test_infer_from_regions_empty_regions_raises(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """infer_from_regions raises ValueError for empty regions."""
        with pytest.raises(ValueError, match="regions must not be empty"):
            roi_engine.infer_from_regions(sample_frame, [])


# ---------------------------------------------------------------------------
# ROIInferenceEngine — multiple ROI tests
# ---------------------------------------------------------------------------

class TestROIInferenceEngineMultipleROIs:
    """Tests for multiple ROI inference."""

    def test_infer_rois_two_rois(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """infer_rois works with two ROIs."""
        region_a = Region(x=0, y=0, width=32, height=32)
        region_b = Region(x=32, y=32, width=32, height=32)

        extracted_a = extractor.extract(sample_frame, region_a)
        extracted_b = extractor.extract(sample_frame, region_b)

        results = roi_engine.infer_rois([extracted_a, extracted_b])

        assert len(results) == 2
        assert results[0].first_output is not None
        assert results[1].first_output is not None

    def test_infer_rois_preserves_ordering(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Results are returned in the same order as input."""
        regions = [
            Region(x=0, y=0, width=16, height=16),
            Region(x=16, y=0, width=16, height=16),
            Region(x=32, y=0, width=16, height=16),
            Region(x=0, y=16, width=16, height=16),
        ]

        extracted = extractor.extract_many(sample_frame, regions)
        results = roi_engine.infer_rois(extracted)

        assert len(results) == 4
        for i, (result, region) in enumerate(zip(results, regions)):
            assert result.region == region

    def test_infer_rois_identical_regions(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Same region supplied twice produces two independent results."""
        region = Region(x=8, y=8, width=32, height=32)
        extracted = extractor.extract_many(sample_frame, [region, region])

        results = roi_engine.infer_rois(extracted)

        assert len(results) == 2
        np.testing.assert_array_equal(results[0].first_output, results[1].first_output)

    def test_infer_rois_adjacent_regions(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Adjacent (non-overlapping) regions work correctly."""
        regions = [
            Region(x=0, y=0, width=32, height=32),
            Region(x=32, y=0, width=32, height=32),
        ]
        extracted = extractor.extract_many(sample_frame, regions)
        results = roi_engine.infer_rois(extracted)

        assert len(results) == 2
        assert results[0].region.x == 0
        assert results[1].region.x == 32

    def test_infer_rois_overlapping_regions(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Overlapping regions are handled independently."""
        regions = [
            Region(x=0, y=0, width=40, height=40),
            Region(x=20, y=20, width=40, height=40),
        ]
        extracted = extractor.extract_many(sample_frame, regions)
        results = roi_engine.infer_rois(extracted)

        assert len(results) == 2
        assert results[0].first_output is not None
        assert results[1].first_output is not None

    def test_infer_rois_different_sizes(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """ROIs of different sizes are handled via preprocessing."""
        regions = [
            Region(x=0, y=0, width=16, height=16),
            Region(x=0, y=0, width=32, height=32),
            Region(x=0, y=0, width=48, height=48),
        ]
        extracted = extractor.extract_many(sample_frame, regions)
        results = roi_engine.infer_rois(extracted)

        assert len(results) == 3
        for result in results:
            assert result.first_output is not None


# ---------------------------------------------------------------------------
# ROIInferenceEngine — infer_from_regions tests
# ---------------------------------------------------------------------------

class TestROIInferenceEngineFromRegions:
    """Tests for the infer_from_regions convenience method."""

    def test_infer_from_regions_basic(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """infer_from_regions extracts and infers in one call."""
        regions = [Region(x=0, y=0, width=32, height=32)]
        results = roi_engine.infer_from_regions(sample_frame, regions)

        assert len(results) == 1
        assert results[0].first_output is not None
        assert results[0].region == regions[0]
        assert results[0].roi_index == 0

    def test_infer_from_regions_multiple(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """infer_from_regions works with multiple regions."""
        regions = [
            Region(x=0, y=0, width=32, height=32),
            Region(x=16, y=16, width=32, height=32),
        ]
        results = roi_engine.infer_from_regions(sample_frame, regions)

        assert len(results) == 2
        assert results[0].roi_index == 0
        assert results[1].roi_index == 1

    def test_infer_from_regions_extraction_timing(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """Extraction timing is recorded separately."""
        regions = [Region(x=0, y=0, width=32, height=32)]
        results = roi_engine.infer_from_regions(sample_frame, regions)

        assert results[0].extraction_time_s >= 0
        # total time should include extraction
        assert results[0].total_time_s >= results[0].total_inference_time_s

    def test_infer_from_regions_preserves_coordinates(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """Original region coordinates are preserved through the pipeline."""
        regions = [
            Region(x=10, y=20, width=32, height=32),
            Region(x=40, y=50, width=24, height=24),
        ]
        results = roi_engine.infer_from_regions(sample_frame, regions)

        assert results[0].region.x == 10
        assert results[0].region.y == 20
        assert results[1].region.x == 40
        assert results[1].region.y == 50

    def test_infer_from_regions_custom_extractor(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """infer_from_regions accepts a custom extractor."""
        custom_extractor = ROIExtractor()
        regions = [Region(x=0, y=0, width=32, height=32)]
        results = roi_engine.infer_from_regions(sample_frame, regions, extractor=custom_extractor)

        assert len(results) == 1
        assert results[0].first_output is not None


# ---------------------------------------------------------------------------
# ROIInferenceEngine — property tests
# ---------------------------------------------------------------------------

class TestROIInferenceEngineProperties:
    """Tests for ROIInferenceEngine properties."""

    def test_inference_engine_property(self, roi_engine: ROIInferenceEngine, loaded_engine: InferenceEngine):
        """inference_engine property returns the underlying engine."""
        assert roi_engine.inference_engine is loaded_engine

    def test_model_info_property(self, roi_engine: ROIInferenceEngine):
        """model_info property delegates to underlying engine."""
        info = roi_engine.model_info
        assert info is not None
        assert info.model_name == "test_model"


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestROIInferenceIntegration:
    """Integration tests for the full ROI inference pipeline."""

    def test_full_pipeline(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Full pipeline: frame -> region -> extract -> infer -> result."""
        region = Region(x=8, y=8, width=48, height=48)
        extracted = extractor.extract(sample_frame, region)
        result = roi_engine.infer_roi(extracted)

        # Verify all components
        assert result.first_output is not None
        assert result.first_output.shape == (1, 10)
        assert result.region == region
        assert result.preprocessing_time_s >= 0
        assert result.inference_time_s > 0
        assert result.total_time_s > 0
        assert result.model_name == "test_model"
        assert result.device == "CPU"

    def test_full_pipeline_from_regions(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray
    ):
        """Full pipeline using infer_from_regions convenience method."""
        regions = [
            Region(x=0, y=0, width=32, height=32),
            Region(x=32, y=32, width=32, height=32),
        ]
        results = roi_engine.infer_from_regions(sample_frame, regions)

        assert len(results) == 2
        for result in results:
            assert result.first_output is not None
            assert result.first_output.shape == (1, 10)
            assert result.inference_time_s > 0

    def test_full_frame_still_works(
        self, loaded_engine: InferenceEngine, sample_frame: np.ndarray
    ):
        """Existing full-frame inference path remains intact."""
        result = loaded_engine.infer(sample_frame)
        assert result.first_output is not None
        assert result.first_output.shape == (1, 10)
        assert result.model_name == "test_model"

    def test_source_frame_not_mutated(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """Extraction does not modify the source frame."""
        original = sample_frame.copy()
        region = Region(x=0, y=0, width=32, height=32)
        roi_engine.infer_from_regions(sample_frame, [region])
        np.testing.assert_array_equal(sample_frame, original)

    def test_consistency_with_full_frame(
        self, roi_engine: ROIInferenceEngine, sample_frame: np.ndarray, extractor: ROIExtractor
    ):
        """ROI covering full frame produces valid output (not necessarily identical to full-frame
        due to preprocessing differences, but both should succeed)."""
        h, w = sample_frame.shape[:2]
        region = Region(x=0, y=0, width=w, height=h)

        # ROI path
        extracted = extractor.extract(sample_frame, region)
        roi_result = roi_engine.infer_roi(extracted)

        # Full-frame path
        full_result = roi_engine.inference_engine.infer(sample_frame)

        # Both should produce valid outputs with the same shape
        assert roi_result.first_output.shape == full_result.first_output.shape
