"""Unit tests for foveaedge.benchmark.result — BenchmarkResult."""

from __future__ import annotations

import json

import pytest

from foveaedge.benchmark.config import BenchmarkConfig
from foveaedge.benchmark.result import BenchmarkResult
from foveaedge.benchmark.stats import TimingStats
from foveaedge.environment import EnvironmentSnapshot


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    @pytest.fixture
    def result(self) -> BenchmarkResult:
        """Create a minimal BenchmarkResult."""
        config = BenchmarkConfig(model_path="m.xml", warmup_frames=5, measured_frames=20)
        env = EnvironmentSnapshot(openvino_version="2024.6.0", numpy_version="1.26.4")
        inf_stats = TimingStats(count=20, mean=0.01, std=0.001, min=0.005, max=0.02, p50=0.01, p90=0.015, p95=0.018, p99=0.02)
        total_stats = TimingStats(count=20, mean=0.015, std=0.002, min=0.008, max=0.025, p50=0.015, p90=0.02, p95=0.022, p99=0.025)
        return BenchmarkResult(
            config=config,
            environment=env,
            inference_stats=inf_stats,
            total_stats=total_stats,
            fps=66.67,
            warmup_frames=5,
            measured_frames=20,
            successful_frames=25,
            failed_frames=0,
        )

    def test_to_dict_structure(self, result: BenchmarkResult):
        """to_dict has correct top-level keys."""
        d = result.to_dict()
        assert "experiment" in d
        assert "configuration" in d
        assert "environment" in d
        assert "timing" in d
        assert "statistics" in d

    def test_to_json_valid(self, result: BenchmarkResult):
        """to_json produces valid JSON."""
        j = result.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_to_json_has_experiment(self, result: BenchmarkResult):
        """JSON has experiment section with timing fields."""
        d = json.loads(result.to_json())
        assert d["experiment"]["warmup_frames"] == 5
        assert d["experiment"]["measured_frames"] == 20
        assert d["experiment"]["successful_frames"] == 25

    def test_to_json_has_timing(self, result: BenchmarkResult):
        """JSON has timing section with inference stats."""
        d = json.loads(result.to_json())
        assert d["timing"]["inference"] is not None
        assert d["timing"]["inference"]["mean"] == 0.01

    def test_to_json_has_fps(self, result: BenchmarkResult):
        """JSON has FPS in statistics."""
        d = json.loads(result.to_json())
        assert d["statistics"]["fps"] == pytest.approx(66.67)

    def test_summary(self, result: BenchmarkResult):
        """summary returns a string."""
        s = result.summary()
        assert isinstance(s, str)
        assert "FoveaEdge" in s
        assert "FPS" in s

    def test_empty_result(self):
        """Empty result still serializes."""
        config = BenchmarkConfig(model_path="m.xml")
        env = EnvironmentSnapshot()
        result = BenchmarkResult(config=config, environment=env)
        d = result.to_dict()
        assert d["timing"]["inference"] is None
        assert d["statistics"]["fps"] == 0.0
