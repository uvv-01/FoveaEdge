"""Synchronous OpenVINO inference engine.

Provides InferenceEngine for performing inference on images with
per-stage timing and structured results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openvino.runtime import CompiledModel, InferRequest

from foveaedge.model import ModelInfo, ModelLoader


@dataclass
class InferenceResult:
    """Structured result from a single inference pass.

    Attributes:
        output_tensors: Dict mapping output name to numpy array.
        preprocessing_time_s: Time spent preprocessing (seconds).
        inference_time_s: Time spent in OpenVINO inference (seconds).
        postprocessing_time_s: Time spent extracting outputs (seconds).
        total_time_s: Total wall-clock time for the full pipeline (seconds).
        input_shape: Shape of the preprocessed input fed to the model.
        model_name: Name of the model used.
        device: Device used for inference.
    """

    output_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    preprocessing_time_s: float = 0.0
    inference_time_s: float = 0.0
    postprocessing_time_s: float = 0.0
    total_time_s: float = 0.0
    input_shape: tuple[int, ...] = ()
    model_name: str = ""
    device: str = ""

    @property
    def output_names(self) -> list[str]:
        """Return names of output tensors."""
        return list(self.output_tensors.keys())

    @property
    def first_output(self) -> np.ndarray | None:
        """Return the first output tensor, or None."""
        if self.output_tensors:
            return next(iter(self.output_tensors.values()))
        return None

    def __repr__(self) -> str:
        return (
            f"InferenceResult(outputs={len(self.output_tensors)}, "
            f"inference={self.inference_time_s:.4f}s, total={self.total_time_s:.4f}s)"
        )


def preprocess_image(
    image: np.ndarray,
    target_shape: tuple[int, ...],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Preprocess an image for model input.

    Handles resize, normalization to [0, 1], and HWC -> NCHW conversion.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W).
        target_shape: Target shape as (N, C, H, W) or (C, H, W).
        dtype: Output data type.

    Returns:
        Preprocessed image array matching the model's expected input shape.
    """
    # Determine target H, W from shape
    if len(target_shape) == 4:
        # NCHW
        n, c, h, w = target_shape
    elif len(target_shape) == 3:
        # CHW
        c, h, w = target_shape
        n = 1
    else:
        raise ValueError(f"Unexpected target shape: {target_shape}")

    # Ensure image is 2D or 3D
    if image.ndim == 2:
        # Grayscale -> add channel dim
        image = image[:, :, np.newaxis] if c == 1 else np.stack([image] * c, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1 and c > 1:
        # Single channel -> repeat
        image = np.concatenate([image] * c, axis=-1)

    # Resize only if image is HWC or HW (not already NCHW)
    needs_resize = False
    if image.ndim == 3 and image.shape[2] in (1, 3):
        # HWC format - check if H,W match
        if image.shape[0] != h or image.shape[1] != w:
            needs_resize = True
    elif image.ndim == 2:
        # HW format - check if H,W match
        if image.shape[0] != h or image.shape[1] != w:
            needs_resize = True
    elif image.ndim == 4 and image.shape == (n, c, h, w):
        # Already NCHW with correct shape
        pass
    else:
        needs_resize = True

    if needs_resize:
        try:
            import cv2

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image

            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize((w, h), Image.BILINEAR)
            image = np.array(img_pil)

    # Ensure correct channel count
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    # HWC -> CHW
    if image.ndim == 3 and image.shape[2] in (1, 3):
        image = np.transpose(image, (2, 0, 1))

    # Normalize to [0, 1]
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        image = image.astype(np.float32) / float(max_val)
    elif image.dtype != dtype:
        image = image.astype(dtype)

    # Add batch dimension if needed
    if len(target_shape) == 4 and image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    return image.astype(dtype)


class InferenceEngine:
    """Synchronous OpenVINO inference engine.

    Responsibilities:
        - Receive an image/frame
        - Preprocess it to match model input
        - Perform OpenVINO inference
        - Collect output tensors
        - Return a structured InferenceResult
        - Measure preprocessing, inference, postprocessing, and total time

    Usage:
        from foveaedge.model import ModelLoader
        from foveaedge.inference.engine import InferenceEngine

        loader = ModelLoader()
        loader.load_and_compile("model.xml", device="CPU")
        engine = InferenceEngine(loader)
        result = engine.infer(image)
    """

    def __init__(self, model_loader: ModelLoader) -> None:
        """Initialize with a loaded and compiled ModelLoader.

        Args:
            model_loader: A ModelLoader with a compiled model.

        Raises:
            RuntimeError: If no compiled model is available.
        """
        if model_loader.compiled_model is None:
            raise RuntimeError("ModelLoader must have a compiled model. Call load_and_compile() first.")

        self._loader = model_loader
        self._compiled: CompiledModel = model_loader.compiled_model
        self._model_info: ModelInfo = model_loader.model_info
        self._infer_request: InferRequest = self._compiled.create_infer_request()

    @property
    def model_info(self) -> ModelInfo:
        """Metadata about the loaded model."""
        return self._model_info

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Expected input shape of the model."""
        if self._model_info.input_tensors:
            return self._model_info.input_tensors[0].shape
        return ()

    @property
    def device(self) -> str:
        """Device used for inference."""
        return self._model_info.compiled_device or self._model_info.requested_device

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for model input.

        Args:
            image: Input image as numpy array.

        Returns:
            Preprocessed array matching model input shape.
        """
        target_shape = self.input_shape
        input_info = self._model_info.input_tensors[0]
        # Determine dtype from model
        dtype_str = input_info.element_type
        if "f32" in dtype_str or "float" in dtype_str:
            dtype = np.float32
        elif "f16" in dtype_str:
            dtype = np.float16
        elif "i8" in dtype_str or "int8" in dtype_str:
            dtype = np.int8
        elif "u8" in dtype_str or "uint8" in dtype_str:
            dtype = np.uint8
        else:
            dtype = np.float32

        return preprocess_image(image, target_shape, dtype)

    def infer(self, image: np.ndarray) -> InferenceResult:
        """Run synchronous inference on an image.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W).

        Returns:
            InferenceResult with outputs and timing information.
        """
        total_start = time.perf_counter()

        # Preprocessing
        pre_start = time.perf_counter()
        preprocessed = self.preprocess(image)
        pre_end = time.perf_counter()
        preprocessing_time = pre_end - pre_start

        # Inference
        inf_start = time.perf_counter()
        input_name = self._model_info.input_tensors[0].name if self._model_info.input_tensors else "input"
        self._infer_request.infer({input_name: preprocessed})
        inf_end = time.perf_counter()
        inference_time = inf_end - inf_start

        # Postprocessing (output extraction)
        post_start = time.perf_counter()
        output_tensors = {}
        for i, output_info in enumerate(self._model_info.output_tensors):
            output_tensors[output_info.name] = self._infer_request.get_output_tensor(i).data.copy()
        post_end = time.perf_counter()
        postprocessing_time = post_end - post_start

        total_end = time.perf_counter()

        return InferenceResult(
            output_tensors=output_tensors,
            preprocessing_time_s=preprocessing_time,
            inference_time_s=inference_time,
            postprocessing_time_s=postprocessing_time,
            total_time_s=total_end - total_start,
            input_shape=preprocessed.shape,
            model_name=self._model_info.model_name,
            device=self.device,
        )
