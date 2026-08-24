"""Statistics module for benchmark timing analysis.

Provides percentile calculation, FPS, and summary statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    """Statistics computed from a set of timing measurements.

    Attributes:
        count: Number of samples.
        mean: Mean value.
        std: Standard deviation.
        min: Minimum value.
        max: Maximum value.
        p50: 50th percentile (median).
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
    """

    count: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


def compute_timing_stats(times: list[float]) -> TimingStats:
    """Compute timing statistics from a list of measurements.

    Args:
        times: List of timing measurements in seconds.

    Returns:
        TimingStats with computed statistics.

    Raises:
        ValueError: If times is empty.
    """
    if not times:
        raise ValueError("Cannot compute statistics from empty list")

    arr = np.array(times, dtype=np.float64)
    n = len(arr)

    if n == 1:
        return TimingStats(
            count=1,
            mean=float(arr[0]),
            std=0.0,
            min=float(arr[0]),
            max=float(arr[0]),
            p50=float(arr[0]),
            p90=float(arr[0]),
            p95=float(arr[0]),
            p99=float(arr[0]),
        )

    return TimingStats(
        count=n,
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def compute_fps(times: list[float]) -> float:
    """Compute frames per second from timing measurements.

    Args:
        times: List of per-frame total times in seconds.

    Returns:
        FPS value (frames / total_time). Returns 0.0 if total time is zero.
    """
    if not times:
        return 0.0
    total = sum(times)
    if total <= 0:
        return 0.0
    return len(times) / total
