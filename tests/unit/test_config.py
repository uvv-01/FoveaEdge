"""Tests for foveaedge.config."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from foveaedge.config import (
    Config,
    DeviceConfig,
    CaptureConfig,
    PeripheralConfig,
    ROIConfig,
    SchedulerConfig,
    InferenceConfig,
    TrackingConfig,
    TelemetryConfig,
)


class TestDeviceConfig:
    def test_defaults(self):
        cfg = DeviceConfig()
        assert cfg.device == "CPU"
        assert cfg.performance_hint == "LATENCY"
        assert cfg.streams == 0
        assert cfg.enable_gpu is True

    def test_custom(self):
        cfg = DeviceConfig(device="GPU", performance_hint="THROUGHPUT")
        assert cfg.device == "GPU"
        assert cfg.performance_hint == "THROUGHPUT"


class TestCaptureConfig:
    def test_defaults(self):
        cfg = CaptureConfig()
        assert cfg.width == 1920
        assert cfg.height == 1080
        assert cfg.fps == 30.0
        assert cfg.max_frames == 0


class TestROIConfig:
    def test_defaults(self):
        cfg = ROIConfig()
        assert cfg.roi_width == 640
        assert cfg.roi_height == 640
        assert cfg.max_rois == 8
        assert 0 < cfg.merge_iou_threshold < 1

    def test_min_max(self):
        cfg = ROIConfig(min_roi_size=10, max_rois=16)
        assert cfg.min_roi_size == 10
        assert cfg.max_rois == 16


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.max_inferences_per_frame == 4
        assert cfg.target_frame_latency_ms == 33.0

    def test_scoring_weights_sum(self):
        cfg = SchedulerConfig()
        total = (
            cfg.weight_motion
            + cfg.weight_persistence
            + cfg.weight_novelty
            + cfg.weight_area_penalty
            + cfg.weight_overlap_penalty
        )
        assert abs(total - 1.0) < 0.01


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.device.device == "CPU"
        assert cfg.capture.width == 1920
        assert cfg.roi.max_rois == 8

    def test_to_dict(self):
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "device" in d
        assert d["device"]["device"] == "CPU"

    def test_from_dict(self):
        data = {
            "device": {"device": "GPU", "performance_hint": "THROUGHPUT"},
            "capture": {"width": 640, "height": 480},
        }
        cfg = Config.from_dict(data)
        assert cfg.device.device == "GPU"
        assert cfg.capture.width == 640
        # Unspecified fields use defaults
        assert cfg.roi.max_rois == 8

    def test_roundtrip_dict(self):
        original = Config()
        d = original.to_dict()
        restored = Config.from_dict(d)
        assert original.to_dict() == restored.to_dict()

    def test_json_roundtrip(self):
        original = Config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            original.to_json(path)
            restored = Config.from_json(path)
            assert original.to_dict() == restored.to_dict()
        finally:
            Path(path).unlink()

    def test_yaml_roundtrip(self):
        original = Config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            original.to_yaml(path)
            restored = Config.from_yaml(path)
            assert original.to_dict() == restored.to_dict()
        finally:
            Path(path).unlink()

    def test_to_json_string(self):
        cfg = Config()
        text = cfg.to_json()
        data = json.loads(text)
        assert data["device"]["device"] == "CPU"

    def test_to_yaml_string(self):
        cfg = Config()
        text = cfg.to_yaml()
        data = yaml.safe_load(text)
        assert data["device"]["device"] == "CPU"
