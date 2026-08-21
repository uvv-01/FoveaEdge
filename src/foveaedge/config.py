"""Central configuration for FoveaEdge.

Provides a dataclass-based configuration system with YAML serialization.
All subsystems read their configuration from a single Config instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DeviceConfig:
    """OpenVINO device selection and hints."""
    device: str = "CPU"
    performance_hint: str = "LATENCY"
    streams: int = 0
    enable_gpu: bool = True

@dataclass
class CaptureConfig:
    """Video input source configuration."""
    source: str = "0"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    max_frames: int = 0
    shutdown_timeout: float = 5.0

@dataclass
class PeripheralConfig:
    """Low-cost event / motion detection parameters."""
    bg_history: int = 500
    bg_threshold: int = 500
    min_motion_area_ratio: float = 0.001
    blur_kernel: int = 21
    dilate_iterations: int = 3
    morph_kernel: int = 5

@dataclass
class ROIConfig:
    """Region-of-interest generation, scoring, and merging parameters."""
    roi_width: int = 640
    roi_height: int = 640
    roi_padding: int = 32
    min_roi_size: int = 32
    max_rois: int = 8
    min_score: float = 0.1
    merge_iou_threshold: float = 0.4
    max_roi_area_ratio: float = 0.5

@dataclass
class SchedulerConfig:
    """Spatial scheduling and compute budget parameters."""
    max_inferences_per_frame: int = 4
    target_frame_latency_ms: float = 33.0
    roi_reinfer_interval: int = 5
    min_reinfer_interval_ms: float = 100.0
    weight_motion: float = 0.4
    weight_persistence: float = 0.2
    weight_novelty: float = 0.2
    weight_area_penalty: float = 0.1
    weight_overlap_penalty: float = 0.1

@dataclass
class InferenceConfig:
    """OpenVINO inference pipeline settings."""
    model_path: str = "ssd_mobilenet_v2/SSD-MobileNetV2.xml"
    confidence_threshold: float = 0.5
    use_async: bool = True
    async_queue_depth: int = 4
    input_width: int = 300
    input_height: int = 300
    mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

@dataclass
class TrackingConfig:
    """Temporal tracking and state persistence settings."""
    enable_temporal_reuse: bool = True
    max_track_age: int = 10
    confidence_decay: float = 0.9
    reinfer_confidence_threshold: float = 0.3

@dataclass
class TelemetryConfig:
    """Performance metrics and reporting settings."""
    enabled: bool = True
    window_size: int = 30
    report_interval: float = 5.0
    output_dir: str = "docs/progress"

@dataclass
class Config:
    """Root configuration aggregating all subsystem configs."""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    peripheral: PeripheralConfig = field(default_factory=PeripheralConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        sub_configs = {
            "device": DeviceConfig,
            "capture": CaptureConfig,
            "peripheral": PeripheralConfig,
            "roi": ROIConfig,
            "scheduler": SchedulerConfig,
            "inference": InferenceConfig,
            "tracking": TrackingConfig,
            "telemetry": TelemetryConfig,
        }
        kwargs = {}
        for key, klass in sub_configs.items():
            if key in data:
                kwargs[key] = klass(**data[key])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))
