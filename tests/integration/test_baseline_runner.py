"""Integration tests for foveaedge.benchmark.runner — BaselineRunner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveaedge.benchmark.config import BenchmarkConfig
from foveaedge.benchmark.frame_source import SyntheticFrameSource
from foveaedge.benchmark.runner import BaselineRunner

TEST_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "test_model"
TEST_MODEL_XML = TEST_MODEL_DIR / "test_model.xml"


@pytest.fixture
def config() -> BenchmarkConfig:
    """Create a minimal benchmark config using the test model."""
    if not TEST_MODEL_XML.exists():
        pytest.skip("Test model not generated")
    return BenchmarkConfig(
        model_path=str(TEST_MODEL_XML),
        device="CPU",
        warmup_frames=3,
        measured_frames=10,
        input_width=32,
        input_height=32,
        input_channels=3,
        seed=42,
    )


class TestBaselineRunner:
    """Integration tests for BaselineRunner."""

    def test_run_with_synthetic(self, config: BenchmarkConfig):
        """BaselineRunner runs with synthetic frames."""
        runner = BaselineRunner(config)
        result = runner.run()
        assert result.successful_frames == config.total_frames
        assert result.failed_frames == 0

    def test_warmup_excluded(self, config: BenchmarkConfig):
        """Warmup frames are excluded from timing stats."""
        runner = BaselineRunner(config)
        result = runner.run()
        assert result.warmup_frames == config.warmup_frames
        assert result.measured_frames == config.measured_frames
        # Stats should be computed for measured frames only
        assert result.inference_stats is not None
        assert result.inference_stats.count == config.measured_frames

    def test_timing_non_negative(self, config: BenchmarkConfig):
        """All timing values are non-negative."""
        runner = BaselineRunner(config)
        result = runner.run()
        assert result.preprocessing_stats is not None
        assert result.preprocessing_stats.min >= 0
        assert result.inference_stats is not None
        assert result.inference_stats.min >= 0
        assert result.total_stats is not None
        assert result.total_stats.min >= 0

    def test_fps_positive(self, config: BenchmarkConfig):
        """FPS is positive when frames are processed."""
        runner = BaselineRunner(config)
        result = runner.run()
        assert result.fps > 0

    def test_json_output(self, config: BenchmarkConfig):
        """Result can be serialized to JSON."""
        import json

        runner = BaselineRunner(config)
        result = runner.run()
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["configuration"]["device"] == "CPU"
        assert parsed["statistics"]["fps"] > 0

    def test_summary_output(self, config: BenchmarkConfig):
        """Result summary is a readable string."""
        runner = BaselineRunner(config)
        result = runner.run()
        s = result.summary()
        assert "FoveaEdge" in s
        assert "FPS" in s

    def test_custom_frame_source(self, config: BenchmarkConfig):
        """BaselineRunner accepts custom frame source."""
        src = SyntheticFrameSource(count=config.total_frames, width=32, height=32, channels=3, seed=99)
        runner = BaselineRunner(config)
        result = runner.run(frame_source=src)
        assert result.successful_frames == config.total_frames

    def test_deterministic_results(self, config: BenchmarkConfig):
        """Same config produces structurally similar results."""
        runner = BaselineRunner(config)
        r1 = runner.run()
        r2 = runner.run()
        # Both should have same structure
        assert r1.warmup_frames == r2.warmup_frames
        assert r1.measured_frames == r2.measured_frames
        assert r1.config.seed == r2.config.seed

    def test_invalid_config(self):
        """Invalid config raises ValueError."""
        config = BenchmarkConfig()  # missing model_path
        runner = BaselineRunner(config)
        with pytest.raises(ValueError, match="Invalid config"):
            runner.run()
