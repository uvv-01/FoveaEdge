"""OpenVINO model loading, metadata, and compilation.

Provides ModelLoader for loading and compiling OpenVINO models,
ModelInfo for model metadata, and TensorInfo for tensor metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openvino.runtime import CompiledModel, Core, Model


@dataclass(frozen=True)
class TensorInfo:
    """Metadata about a single input or output tensor.

    Attributes:
        name: Tensor name as reported by OpenVINO.
        shape: Shape tuple (e.g. (1, 3, 224, 224)).
        element_type: Data type string (e.g. 'f32', 'i8').
        layout: Layout string if available (e.g. 'NCHW').
    """

    name: str
    shape: tuple[int, ...]
    element_type: str
    layout: str = ""

    def __repr__(self) -> str:
        return f"TensorInfo(name={self.name!r}, shape={self.shape}, type={self.element_type})"


@dataclass(frozen=True)
class ModelInfo:
    """Metadata extracted from an OpenVINO model.

    Attributes:
        model_path: Path to the model file.
        model_name: Descriptive name derived from the file path.
        input_tensors: List of input tensor metadata.
        output_tensors: List of output tensor metadata.
        requested_device: The device string requested for compilation.
        compiled_device: The actual device used after compilation (if compiled).
    """

    model_path: str
    model_name: str
    input_tensors: list[TensorInfo] = field(default_factory=list)
    output_tensors: list[TensorInfo] = field(default_factory=list)
    requested_device: str = ""
    compiled_device: str = ""

    @property
    def input_names(self) -> list[str]:
        """Return names of all input tensors."""
        return [t.name for t in self.input_tensors]

    @property
    def output_names(self) -> list[str]:
        """Return names of all output tensors."""
        return [t.name for t in self.output_tensors]

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        """Return shapes of all input tensors."""
        return [t.shape for t in self.input_tensors]

    @property
    def output_shapes(self) -> list[tuple[int, ...]]:
        """Return shapes of all output tensors."""
        return [t.shape for t in self.output_tensors]

    def __repr__(self) -> str:
        return (
            f"ModelInfo(name={self.model_name!r}, inputs={len(self.input_tensors)}, "
            f"outputs={len(self.output_tensors)}, device={self.compiled_device or self.requested_device})"
        )


def _extract_tensor_info(name: str, ov_tensor: Any) -> TensorInfo:
    """Extract TensorInfo from an OpenVINO tensor description."""
    shape = tuple(ov_tensor.shape)
    element_type = str(ov_tensor.element_type)
    layout = str(ov_tensor.layout) if hasattr(ov_tensor, "layout") else ""
    return TensorInfo(name=name, shape=shape, element_type=element_type, layout=layout)


def _extract_model_info(model: Model, model_path: str, device: str = "") -> ModelInfo:
    """Extract ModelInfo from an OpenVINO Model object."""
    model_name = Path(model_path).stem

    inputs = []
    for idx in range(len(model.inputs)):
        tensor = model.inputs[idx]
        name = tensor.any_name
        inputs.append(_extract_tensor_info(name, tensor))

    outputs = []
    for idx in range(len(model.outputs)):
        tensor = model.outputs[idx]
        name = tensor.any_name
        outputs.append(_extract_tensor_info(name, tensor))

    return ModelInfo(
        model_path=str(model_path),
        model_name=model_name,
        input_tensors=inputs,
        output_tensors=outputs,
        requested_device=device,
    )


class ModelLoader:
    """Load and compile OpenVINO models.

    Responsibilities:
        - Accept an OpenVINO model path (.xml, .onnx, etc.)
        - Load the model via OpenVINO Core
        - Compile the model for a requested device
        - Expose model metadata via ModelInfo
        - Validate requested devices
        - Provide useful errors for invalid models/devices

    Usage:
        loader = ModelLoader()
        info = loader.load("model.xml", device="CPU")
        compiled = loader.compiled_model
    """

    def __init__(self) -> None:
        """Initialize with a fresh OpenVINO Core."""
        self._core = Core()
        self._model: Model | None = None
        self._compiled_model: CompiledModel | None = None
        self._model_info: ModelInfo | None = None

    @property
    def core(self) -> Core:
        """Access the underlying OpenVINO Core."""
        return self._core

    @property
    def model(self) -> Model | None:
        """The loaded (but not compiled) model, or None."""
        return self._model

    @property
    def compiled_model(self) -> CompiledModel | None:
        """The compiled model, or None if not yet compiled."""
        return self._compiled_model

    @property
    def model_info(self) -> ModelInfo | None:
        """Metadata about the loaded model, or None."""
        return self._model_info

    def available_devices(self) -> list[str]:
        """Return list of available OpenVINO devices."""
        return list(self._core.available_devices)

    def validate_device(self, device: str) -> bool:
        """Check if a device is available on this system."""
        available = self.available_devices()
        return device in available

    def load(self, model_path: str | Path, device: str = "CPU") -> ModelInfo:
        """Load an OpenVINO model from a file path.

        Args:
            model_path: Path to the model file (.xml, .onnx, .pb, etc.).
            device: Target device for compilation (default: "CPU").

        Returns:
            ModelInfo with extracted metadata.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the model cannot be loaded or device is unavailable.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not self.validate_device(device):
            available = self.available_devices()
            raise RuntimeError(
                f"Device '{device}' is not available. Available devices: {available}"
            )

        try:
            self._model = self._core.read_model(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

        self._model_info = _extract_model_info(self._model, str(model_path), device)
        return self._model_info

    def compile(self, device: str | None = None) -> CompiledModel:
        """Compile the loaded model for a device.

        Args:
            device: Target device. If None, uses the device from the last load() call.

        Returns:
            The compiled OpenVINO model.

        Raises:
            RuntimeError: If no model is loaded or compilation fails.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        if device is None:
            device = self._model_info.requested_device if self._model_info else "CPU"

        if not self.validate_device(device):
            available = self.available_devices()
            raise RuntimeError(
                f"Device '{device}' is not available. Available devices: {available}"
            )

        try:
            self._compiled_model = self._core.compile_model(self._model, device)
        except Exception as e:
            raise RuntimeError(f"Failed to compile model for device '{device}': {e}") from e

        # Update model info with the actual compiled device
        if self._model_info:
            self._model_info = ModelInfo(
                model_path=self._model_info.model_path,
                model_name=self._model_info.model_name,
                input_tensors=self._model_info.input_tensors,
                output_tensors=self._model_info.output_tensors,
                requested_device=self._model_info.requested_device,
                compiled_device=device,
            )

        return self._compiled_model

    def load_and_compile(self, model_path: str | Path, device: str = "CPU") -> ModelInfo:
        """Load and compile a model in one step.

        Args:
            model_path: Path to the model file.
            device: Target device (default: "CPU").

        Returns:
            ModelInfo with metadata including compiled device.
        """
        self.load(model_path, device)
        self.compile(device)
        return self._model_info
