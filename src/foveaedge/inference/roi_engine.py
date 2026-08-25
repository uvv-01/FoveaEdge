"""ROI inference engine for FoveaEdge.

Provides ROIInferenceEngine for running OpenVINO inference on extracted
regions of interest, and ROIInferenceResult for representing results
with their original spatial metadata.

This module connects the ROI extraction layer (Day 5) to the existing
OpenVINO inference infrastructure (Day 2) while preserving coordinate
information needed for future result mapping.

Coordinate convention:
    - Origin: top-left corner of the image
    - x increases rightward, y increases downward
    - ROIInferenceResult.region always refers to the original frame
      coordinates, not the crop-local coordinates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from foveaedge.inference.engine import InferenceEngine, InferenceResult
from foveaedge.regions.extraction import ExtractedROI, ROIExtractor
from foveaedge.regions.region import Region


@dataclass
class ROIInferenceResult:
    """Result of inference on a single extracted ROI.

    Attributes:
        inference_result: The underlying InferenceResult from OpenVINO.
        region: The Region (in original frame coordinates) that this
                ROI was extracted from.
        extraction_time_s: Time spent extracting the ROI from the frame
                           (seconds). This is separate from inference time.
        roi_index: Optional index for ordering/grouping multiple results.
    """

    inference_result: InferenceResult
    region: Region
    extraction_time_s: float = 0.0
    roi_index: int | None = None

    @property
    def output_tensors(self) -> dict[str, np.ndarray]:
        """Delegate to underlying inference result output tensors."""
        return self.inference_result.output_tensors

    @property
    def first_output(self) -> np.ndarray | None:
        """Delegate to underlying inference result first output."""
        return self.inference_result.first_output

    @property
    def preprocessing_time_s(self) -> float:
        """Preprocessing time from the underlying inference result."""
        return self.inference_result.preprocessing_time_s

    @property
    def inference_time_s(self) -> float:
        """Inference time from the underlying inference result."""
        return self.inference_result.inference_time_s

    @property
    def postprocessing_time_s(self) -> float:
        """Postprocessing time from the underlying inference result."""
        return self.inference_result.postprocessing_time_s

    @property
    def total_inference_time_s(self) -> float:
        """Total time from the underlying inference result."""
        return self.inference_result.total_time_s

    @property
    def total_time_s(self) -> float:
        """End-to-end time including extraction, preprocessing, inference,
        and postprocessing."""
        return self.extraction_time_s + self.inference_result.total_time_s

    @property
    def model_name(self) -> str:
        """Model name from the underlying inference result."""
        return self.inference_result.model_name

    @property
    def device(self) -> str:
        """Device from the underlying inference result."""
        return self.inference_result.device

    def __repr__(self) -> str:
        idx_str = f", roi_index={self.roi_index}" if self.roi_index is not None else ""
        return (
            f"ROIInferenceResult(region={self.region}, "
            f"total={self.total_time_s:.4f}s{idx_str})"
        )


class ROIInferenceEngine:
    """Run OpenVINO inference on extracted regions of interest.

    Orchestrates the pipeline:
        ExtractedROI -> preprocessing -> OpenVINO inference -> ROIInferenceResult

    This engine reuses the existing InferenceEngine and ROIExtractor
    without duplicating their logic.

    Usage:
        from foveaedge.model import ModelLoader
        from foveaedge.inference.engine import InferenceEngine
        from foveaedge.inference.roi_engine import ROIInferenceEngine
        from foveaedge.regions import ROIExtractor, Region

        loader = ModelLoader()
        loader.load_and_compile("model.xml", device="CPU")
        engine = InferenceEngine(loader)
        roi_engine = ROIInferenceEngine(engine)

        # Single ROI
        extracted = ROIExtractor().extract(frame, region)
        result = roi_engine.infer_roi(extracted)

        # Multiple ROIs
        extracted_list = ROIExtractor().extract_many(frame, [r1, r2])
        results = roi_engine.infer_rois(extracted_list)
    """

    def __init__(self, inference_engine: InferenceEngine) -> None:
        """Initialize with an existing InferenceEngine.

        Args:
            inference_engine: A compiled InferenceEngine ready for inference.
        """
        self._engine = inference_engine

    @property
    def inference_engine(self) -> InferenceEngine:
        """Access the underlying InferenceEngine."""
        return self._engine

    @property
    def model_info(self):
        """Model metadata from the underlying engine."""
        return self._engine.model_info

    def infer_roi(self, extracted_roi: ExtractedROI) -> ROIInferenceResult:
        """Run inference on a single extracted ROI.

        The ROI image is passed through the existing inference engine's
        preprocessing and inference path. The result preserves the
        original region coordinates.

        Args:
            extracted_roi: An ExtractedROI with image crop and region metadata.

        Returns:
            ROIInferenceResult with inference outputs and region metadata.

        Raises:
            ValueError: If extracted_roi is None or has an invalid image.
        """
        if extracted_roi is None:
            raise ValueError("extracted_roi must not be None")

        if extracted_roi.image is None or extracted_roi.image.size == 0:
            raise ValueError("extracted_roi.image is empty or None")

        # Run inference on the ROI crop
        inference_result = self._engine.infer(extracted_roi.image)

        return ROIInferenceResult(
            inference_result=inference_result,
            region=extracted_roi.region,
            roi_index=extracted_roi.index,
        )

    def infer_rois(
        self, extracted_rois: Sequence[ExtractedROI]
    ) -> list[ROIInferenceResult]:
        """Run inference on multiple extracted ROIs.

        Preserves input ordering. Each ROI is independently inferred.

        Args:
            extracted_rois: Sequence of ExtractedROI objects.

        Returns:
            List of ROIInferenceResult objects, one per input ROI.

        Raises:
            ValueError: If extracted_rois is empty or contains invalid ROIs.
        """
        if not extracted_rois:
            raise ValueError("extracted_rois must not be empty")

        return [self.infer_roi(roi) for roi in extracted_rois]

    def infer_from_regions(
        self,
        frame: np.ndarray,
        regions: list[Region],
        extractor: ROIExtractor | None = None,
    ) -> list[ROIInferenceResult]:
        """Extract ROIs from a frame and run inference on each.

        Convenience method that combines extraction and inference in a
        single call. The extraction and inference timing are recorded
        separately in each ROIInferenceResult.

        Args:
            frame: Source image as a numpy array.
            regions: List of Region objects to extract and infer.
            extractor: Optional ROIExtractor instance. Created if not provided.

        Returns:
            List of ROIInferenceResult objects, one per region.

        Raises:
            ValueError: If frame is None, regions is empty, or extraction fails.
        """
        if frame is None:
            raise ValueError("frame must not be None")

        if not regions:
            raise ValueError("regions must not be empty")

        if extractor is None:
            extractor = ROIExtractor()

        results: list[ROIInferenceResult] = []

        for idx, region in enumerate(regions):
            # Time the extraction
            ext_start = time.perf_counter()
            extracted = extractor.extract(frame, region)
            ext_end = time.perf_counter()
            extraction_time = ext_end - ext_start

            # Run inference
            inference_result = self._engine.infer(extracted.image)

            results.append(
                ROIInferenceResult(
                    inference_result=inference_result,
                    region=extracted.region,
                    extraction_time_s=extraction_time,
                    roi_index=idx,
                )
            )

        return results
