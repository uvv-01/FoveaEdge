"""Benchmark configuration for reproducible experiments.

Defines BenchmarkConfig with warmup, measured frames, seed, device,
and model path settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark experiment.

    Attributes:
        model_path: Path to the OpenVINO model file.
        device: Inference device (default: "CPU").
        warmup_frames: Number of warmup frames (excluded from measurements).
        measured_frames: Number of measured frames.
        input_width: Width of synthetic test frames.
        input_height: Height of synthetic test frames.
        input_channels: Number of channels in test frames.
        seed: Random seed for reproducible synthetic data.
        target_fps: Target FPS for scheduling (informational).
    """

    model_path: str = ""
    device: str = "CPU"
    warmup_frames: int = 10
    measured_frames: int = 100
    input_width: int = 32
    input_height: int = 32
    input_channels: int = 3
    seed: int = 42
    target_fps: int = 30

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if not self.model_path:
            errors.append("model_path is required")
        if self.warmup_frames < 0:
            errors.append("warmup_frames must be >= 0")
        if self.measured_frames <= 0:
            errors.append("measured_frames must be > 0")
        if self.input_width <= 0 or self.input_height <= 0:
            errors.append("input dimensions must be > 0")
        if self.input_channels <= 0:
            errors.append("input_channels must be > 0")
        return errors

    @property
    def total_frames(self) -> int:
        """Total frames including warmup."""
        return self.warmup_frames + self.measured_frames

    def to_dict(self) -> dict:
        """Convert to a dictionary for serialization."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "warmup_frames": self.warmup_frames,
            "measured_frames": self.measured_frames,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_channels": self.input_channels,
            "seed": self.seed,
            "target_fps": self.target_fps,
        }
