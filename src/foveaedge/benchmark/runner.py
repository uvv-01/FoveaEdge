"""Full-frame baseline runner for benchmarking.

Reuses the existing ModelLoader and InferenceEngine to measure
the cost of ordinary full-frame OpenVINO inference.
"""

from __future__ import annotations

import numpy as np

from foveaedge.benchmark.config import BenchmarkConfig
from foveaedge.benchmark.frame_source import FrameSource, SyntheticFrameSource
from foveaedge.benchmark.result import BenchmarkResult
from foveaedge.benchmark.stats import TimingStats, compute_fps, compute_timing_stats
from foveaedge.environment import EnvironmentSnapshot, capture_environment
from foveaedge.inference.engine import InferenceEngine
from foveaedge.model import ModelLoader


class BaselineRunner:
    """Execute full-frame inference benchmark.

    Runs the standard inference pipeline with warmup and measurement phases,
    collecting per-stage timing statistics.

    Usage:
        config = BenchmarkConfig(model_path="model.xml", measured_frames=100)
        runner = BaselineRunner(config)
        result = runner.run()
        print(result.summary())
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the baseline runner.

        Args:
            config: Benchmark configuration.
        """
        self._config = config
        self._environment = capture_environment()

    @property
    def config(self) -> BenchmarkConfig:
        return self._config

    @property
    def environment(self) -> EnvironmentSnapshot:
        return self._environment

    def run(self, frame_source: FrameSource | None = None) -> BenchmarkResult:
        """Execute the full-frame baseline benchmark.

        Args:
            frame_source: Optional custom frame source. If None, creates
                         a SyntheticFrameSource based on config dimensions.

        Returns:
            BenchmarkResult with timing statistics.
        """
        errors = self._config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")

        # Create frame source if not provided
        if frame_source is None:
            frame_source = SyntheticFrameSource(
                count=self._config.total_frames,
                width=self._config.input_width,
                height=self._config.input_height,
                channels=self._config.input_channels,
                seed=self._config.seed,
            )

        # Load model
        loader = ModelLoader()
        loader.load_and_compile(self._config.model_path, self._config.device)
        engine = InferenceEngine(loader)

        # Collect timing data
        pre_times: list[float] = []
        inf_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []
        successful = 0
        failed = 0

        total_frames = self._config.total_frames
        warmup = self._config.warmup_frames
        measured = self._config.measured_frames

        for i in range(total_frames):
            frame = frame_source[i % len(frame_source)]

            try:
                result = engine.infer(frame)
                total_times.append(result.total_time_s)

                if i >= warmup:
                    # Only collect stats for measured frames
                    pre_times.append(result.preprocessing_time_s)
                    inf_times.append(result.inference_time_s)
                    post_times.append(result.postprocessing_time_s)
                    successful += 1
                else:
                    # Warmup frames still count as successful
                    successful += 1

            except Exception:
                failed += 1

        # Compute statistics
        pre_stats = compute_timing_stats(pre_times) if pre_times else None
        inf_stats = compute_timing_stats(inf_times) if inf_times else None
        post_stats = compute_timing_stats(post_times) if post_times else None

        # Total stats from measured frames only
        measured_total = total_times[warmup:] if len(total_times) > warmup else total_times
        total_stats = compute_timing_stats(measured_total) if measured_total else None
        fps = compute_fps(measured_total) if measured_total else 0.0

        return BenchmarkResult(
            config=self._config,
            environment=self._environment,
            preprocessing_stats=pre_stats,
            inference_stats=inf_stats,
            postprocessing_stats=post_stats,
            total_stats=total_stats,
            fps=fps,
            warmup_frames=warmup,
            measured_frames=measured,
            successful_frames=successful,
            failed_frames=failed,
        )
