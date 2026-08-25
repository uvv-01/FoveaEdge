"""Inference package for FoveaEdge.

Provides the synchronous InferenceEngine for OpenVINO inference
and the ROIInferenceEngine for region-based inference.
"""

from foveaedge.inference.engine import InferenceEngine, InferenceResult
from foveaedge.inference.roi_engine import ROIInferenceEngine, ROIInferenceResult

__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "ROIInferenceEngine",
    "ROIInferenceResult",
]
