"""Unit tests for foveaedge.inference.engine — InferenceEngine, preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveaedge.inference.engine import InferenceEngine, InferenceResult, preprocess_image
from foveaedge.model import ModelLoader

TEST_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "test_model"
TEST_MODEL_XML = TEST_MODEL_DIR / "test_model.xml"


# ---------------------------------------------------------------------------
# preprocess_image tests
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    """Tests for the preprocess_image function."""

    def test_resize_and_normalize(self):
        """preprocess_image resizes and normalizes uint8 input."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = preprocess_image(img, (1, 3, 32, 32))
        assert result.shape == (1, 3, 32, 32)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_float32_passthrough(self):
        """preprocess_image handles float32 input."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = preprocess_image(img, (1, 3, 32, 32))
        assert result.shape == (1, 3, 32, 32)
        assert result.dtype == np.float32

    def test_grayscale(self):
        """preprocess_image handles grayscale input."""
        img = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result = preprocess_image(img, (1, 1, 32, 32))
        assert result.shape == (1, 1, 32, 32)

    def test_already_correct_shape(self):
        """preprocess_image handles already-correct input."""
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        result = preprocess_image(img, (1, 3, 32, 32))
        assert result.shape == (1, 3, 32, 32)

    def test_hwc_to_nchw(self):
        """preprocess_image converts HWC to NCHW."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = preprocess_image(img, (1, 3, 32, 32))
        assert result.shape == (1, 3, 32, 32)

    def test_invalid_shape_raises(self):
        """preprocess_image raises ValueError for unexpected target shape."""
        img = np.random.rand(32, 32, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Unexpected target shape"):
            preprocess_image(img, (32,))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InferenceResult tests
# ---------------------------------------------------------------------------


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_creation(self):
        """InferenceResult stores output tensors and timing."""
        result = InferenceResult(
            output_tensors={"out": np.zeros((1, 10))},
            inference_time_s=0.01,
            total_time_s=0.02,
        )
        assert "out" in result.output_tensors
        assert result.inference_time_s == 0.01

    def test_output_names(self):
        """InferenceResult.output_names returns output tensor names."""
        result = InferenceResult(output_tensors={"a": np.zeros(1), "b": np.zeros(1)})
        assert set(result.output_names) == {"a", "b"}

    def test_first_output(self):
        """InferenceResult.first_output returns first tensor."""
        result = InferenceResult(output_tensors={"out": np.zeros((1, 10))})
        assert result.first_output is not None
        assert result.first_output.shape == (1, 10)

    def test_first_output_empty(self):
        """InferenceResult.first_output returns None when empty."""
        result = InferenceResult()
        assert result.first_output is None

    def test_repr(self):
        """InferenceResult repr shows output count and timing."""
        result = InferenceResult(
            output_tensors={"out": np.zeros(1)},
            inference_time_s=0.005,
            total_time_s=0.01,
        )
        r = repr(result)
        assert "1" in r  # 1 output
        assert "0.005" in r


# ---------------------------------------------------------------------------
# InferenceEngine tests
# ---------------------------------------------------------------------------


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    @pytest.fixture
    def engine(self) -> InferenceEngine:
        """Create an InferenceEngine from the test model."""
        if not TEST_MODEL_XML.exists():
            pytest.skip("Test model not generated")
        loader = ModelLoader()
        loader.load_and_compile(str(TEST_MODEL_XML), device="CPU")
        return InferenceEngine(loader)

    def test_init(self, engine: InferenceEngine):
        """InferenceEngine initializes with a compiled model."""
        assert engine is not None

    def test_init_without_compiled_model(self):
        """InferenceEngine raises RuntimeError without compiled model."""
        loader = ModelLoader()
        loader.load(str(TEST_MODEL_XML), device="CPU") if TEST_MODEL_XML.exists() else None
        with pytest.raises(RuntimeError, match="compiled model"):
            InferenceEngine(loader)

    def test_model_info(self, engine: InferenceEngine):
        """InferenceEngine exposes model_info."""
        info = engine.model_info
        assert info.model_name == "test_model"
        assert len(info.input_tensors) > 0

    def test_input_shape(self, engine: InferenceEngine):
        """InferenceEngine.input_shape returns the model's expected input shape."""
        assert engine.input_shape == (1, 3, 32, 32)

    def test_device(self, engine: InferenceEngine):
        """InferenceEngine.device returns the inference device."""
        assert engine.device == "CPU"

    def test_successful_inference(self, engine: InferenceEngine):
        """infer() returns an InferenceResult with output tensors."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        assert isinstance(result, InferenceResult)
        assert len(result.output_tensors) > 0

    def test_output_structure(self, engine: InferenceEngine):
        """infer() produces output with correct shape."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        output = result.first_output
        assert output is not None
        assert output.shape == (1, 10)

    def test_timing_fields(self, engine: InferenceEngine):
        """infer() records non-negative timing values."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        assert result.preprocessing_time_s >= 0
        assert result.inference_time_s >= 0
        assert result.postprocessing_time_s >= 0
        assert result.total_time_s >= 0

    def test_total_time_sums(self, engine: InferenceEngine):
        """total_time_s >= sum of component times."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        component_sum = (
            result.preprocessing_time_s + result.inference_time_s + result.postprocessing_time_s
        )
        assert result.total_time_s >= component_sum - 1e-6  # allow floating point

    def test_preprocessing(self, engine: InferenceEngine):
        """preprocess() returns correctly shaped array."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        preprocessed = engine.preprocess(img)
        assert preprocessed.shape == (1, 3, 32, 32)

    def test_deterministic_output(self, engine: InferenceEngine):
        """Same input produces same output (deterministic model)."""
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        r1 = engine.infer(img)
        r2 = engine.infer(img)
        np.testing.assert_array_equal(r1.first_output, r2.first_output)

    def test_input_shape_in_result(self, engine: InferenceEngine):
        """InferenceResult includes the preprocessed input shape."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        assert result.input_shape == (1, 3, 32, 32)

    def test_model_name_in_result(self, engine: InferenceEngine):
        """InferenceResult includes the model name."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        assert result.model_name == "test_model"

    def test_device_in_result(self, engine: InferenceEngine):
        """InferenceResult includes the device name."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = engine.infer(img)
        assert result.device == "CPU"
