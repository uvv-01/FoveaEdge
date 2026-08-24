"""Unit tests for foveaedge.benchmark.config — BenchmarkConfig."""

from __future__ import annotations

import pytest

from foveaedge.benchmark.config import BenchmarkConfig


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_defaults(self):
        """Default config has reasonable values."""
        config = BenchmarkConfig()
        assert config.device == "CPU"
        assert config.warmup_frames == 10
        assert config.measured_frames == 100
        assert config.seed == 42

    def test_frozen(self):
        """Config is immutable."""
        config = BenchmarkConfig()
        with pytest.raises(AttributeError):
            config.device = "GPU"  # type: ignore[misc]

    def test_validate_empty_model_path(self):
        """Validation fails without model_path."""
        config = BenchmarkConfig()
        errors = config.validate()
        assert any("model_path" in e for e in errors)

    def test_validate_valid_config(self):
        """Validation passes with valid config."""
        config = BenchmarkConfig(model_path="model.xml", measured_frames=50)
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_negative_warmup(self):
        """Validation fails with negative warmup."""
        config = BenchmarkConfig(model_path="m.xml", warmup_frames=-1)
        errors = config.validate()
        assert any("warmup" in e for e in errors)

    def test_validate_zero_measured(self):
        """Validation fails with zero measured frames."""
        config = BenchmarkConfig(model_path="m.xml", measured_frames=0)
        errors = config.validate()
        assert any("measured" in e for e in errors)

    def test_total_frames(self):
        """total_frames sums warmup + measured."""
        config = BenchmarkConfig(warmup_frames=5, measured_frames=20)
        assert config.total_frames == 25

    def test_to_dict(self):
        """to_dict returns serializable dict."""
        config = BenchmarkConfig(model_path="m.xml")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["model_path"] == "m.xml"
        assert d["device"] == "CPU"
        assert "seed" in d
