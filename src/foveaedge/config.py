"""Configuration structures for FoveaEdge inference.

Provides reusable configuration dataclasses for model loading,
inference settings, and device selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model loading and compilation.

    Attributes:
        model_path: Path to the OpenVINO model file (.xml, .onnx, etc.).
        device: Target inference device (default: "CPU").
        num_streams: Number of inference streams (0 = auto).
        cache_dir: Directory for compiled model caching (optional).
    """

    model_path: str = ""
    device: str = "CPU"
    num_streams: int = 0
    cache_dir: str = ""

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if not self.model_path:
            errors.append("model_path is required")
        elif not Path(self.model_path).exists():
            errors.append(f"model_path does not exist: {self.model_path}")
        if not self.device:
            errors.append("device is required")
        return errors


@dataclass
class InferenceConfig:
    """Configuration for inference execution.

    Attributes:
        model: Model configuration.
        target_fps: Target frames per second (for scheduling later).
        timeout_ms: Inference timeout in milliseconds.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    target_fps: int = 30
    timeout_ms: int = 1000
