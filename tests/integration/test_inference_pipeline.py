"""Integration tests for FoveaEdge inference pipeline.

Tests the full flow: load model -> compile -> preprocess -> infer -> result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveaedge.inference.engine import InferenceEngine
from foveaedge.model import ModelLoader

TEST_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "test_model"
TEST_MODEL_XML = TEST_MODEL_DIR / "test_model.xml"
TEST_MODEL_ONNX = TEST_MODEL_DIR / "test_model.onnx"


@pytest.fixture
def loaded_engine() -> InferenceEngine:
    """Load test model, compile on CPU, and return InferenceEngine."""
    if not TEST_MODEL_XML.exists():
        pytest.skip("Test model XML not generated")
    loader = ModelLoader()
    loader.load_and_compile(str(TEST_MODEL_XML), device="CPU")
    return InferenceEngine(loader)


class TestFullInferencePipeline:
    """Integration tests for the full inference pipeline."""

    def test_load_compile_infer(self, loaded_engine: InferenceEngine):
        """Full pipeline: load -> compile -> infer works end-to-end."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = loaded_engine.infer(img)
        assert result.first_output is not None
        assert result.first_output.shape == (1, 10)

    def test_multiple_inferences(self, loaded_engine: InferenceEngine):
        """Multiple sequential inferences work correctly."""
        for i in range(5):
            img = np.ones((32, 32, 3), dtype=np.uint8) * (i * 50)
            result = loaded_engine.infer(img)
            assert result.first_output is not None

    def test_different_image_sizes(self, loaded_engine: InferenceEngine):
        """Different input image sizes are handled via preprocessing."""
        for size in [(32, 32), (64, 64), (128, 128), (224, 224)]:
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            result = loaded_engine.infer(img)
            assert result.first_output is not None
            assert result.first_output.shape == (1, 10)

    def test_timing_is_reasonable(self, loaded_engine: InferenceEngine):
        """Inference timing is non-negative and total >= inference time."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = loaded_engine.infer(img)
        assert result.total_time_s >= result.inference_time_s
        assert result.inference_time_s > 0  # should take some time

    def test_result_metadata(self, loaded_engine: InferenceEngine):
        """InferenceResult includes correct model and device metadata."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = loaded_engine.infer(img)
        assert result.model_name == "test_model"
        assert result.device == "CPU"
        assert result.input_shape == (1, 3, 32, 32)

    def test_onnx_model_pipeline(self):
        """Full pipeline works with ONNX model format."""
        if not TEST_MODEL_ONNX.exists():
            pytest.skip("Test ONNX model not generated")
        loader = ModelLoader()
        loader.load_and_compile(str(TEST_MODEL_ONNX), device="CPU")
        engine = InferenceEngine(loader)
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        assert result.first_output is not None
        assert result.first_output.shape == (1, 10)

    def test_grayscale_input(self, loaded_engine: InferenceEngine):
        """Grayscale input is handled by preprocessing."""
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result = loaded_engine.infer(img)
        assert result.first_output is not None

    def test_float_input(self, loaded_engine: InferenceEngine):
        """Float32 input is handled by preprocessing."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = loaded_engine.infer(img)
        assert result.first_output is not None
