"""Structured benchmark result with machine-readable output.

Provides BenchmarkResult containing experiment metadata, timing statistics,
and JSON serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from foveaedge.benchmark.config import BenchmarkConfig
from foveaedge.benchmark.stats import TimingStats
from foveaedge.environment import EnvironmentSnapshot


@dataclass
class BenchmarkResult:
    """Structured result from a benchmark run.

    Contains separate sections for experiment metadata, configuration,
    environment, timing statistics, and per-stage timing breakdowns.

    Attributes:
        config: Benchmark configuration used.
        environment: Environment snapshot at time of benchmark.
        preprocessing_stats: Timing statistics for preprocessing.
        inference_stats: Timing statistics for inference.
        postprocessing_stats: Timing statistics for postprocessing.
        total_stats: Timing statistics for end-to-end pipeline.
        fps: Measured frames per second.
        warmup_frames: Number of warmup frames executed.
        measured_frames: Number of measured frames.
        successful_frames: Frames that completed without error.
        failed_frames: Frames that encountered errors.
        timestamp: ISO timestamp of the benchmark run.
        notes: Optional notes about the run.
    """

    config: BenchmarkConfig
    environment: EnvironmentSnapshot
    preprocessing_stats: TimingStats | None = None
    inference_stats: TimingStats | None = None
    postprocessing_stats: TimingStats | None = None
    total_stats: TimingStats | None = None
    fps: float = 0.0
    warmup_frames: int = 0
    measured_frames: int = 0
    successful_frames: int = 0
    failed_frames: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to a nested dictionary for JSON serialization."""
        result = {
            "experiment": {
                "timestamp": self.timestamp,
                "warmup_frames": self.warmup_frames,
                "measured_frames": self.measured_frames,
                "successful_frames": self.successful_frames,
                "failed_frames": self.failed_frames,
                "notes": self.notes,
            },
            "configuration": self.config.to_dict(),
            "environment": self.environment.to_dict(),
            "timing": {
                "preprocessing": self.preprocessing_stats.to_dict()
                if self.preprocessing_stats
                else None,
                "inference": self.inference_stats.to_dict()
                if self.inference_stats
                else None,
                "postprocessing": self.postprocessing_stats.to_dict()
                if self.postprocessing_stats
                else None,
                "total": self.total_stats.to_dict() if self.total_stats else None,
            },
            "statistics": {
                "fps": self.fps,
            },
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"FoveaEdge Benchmark Result",
            f"  Timestamp: {self.timestamp}",
            f"  Model: {self.config.model_path}",
            f"  Device: {self.config.device}",
            f"  Warmup: {self.warmup_frames} frames",
            f"  Measured: {self.measured_frames} frames",
            f"  Successful: {self.successful_frames}",
            f"  Failed: {self.failed_frames}",
            f"  FPS: {self.fps:.2f}",
        ]
        if self.inference_stats:
            s = self.inference_stats
            lines.append(f"  Inference: mean={s.mean*1000:.2f}ms p50={s.p50*1000:.2f}ms p99={s.p99*1000:.2f}ms")
        if self.total_stats:
            s = self.total_stats
            lines.append(f"  Total:     mean={s.mean*1000:.2f}ms p50={s.p50*1000:.2f}ms p99={s.p99*1000:.2f}ms")
        return "\n".join(lines)
