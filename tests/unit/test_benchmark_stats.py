"""Unit tests for foveaedge.benchmark.stats — statistics."""

from __future__ import annotations

import pytest

from foveaedge.benchmark.stats import TimingStats, compute_fps, compute_timing_stats


class TestTimingStats:
    """Tests for TimingStats."""

    def test_to_dict(self):
        """TimingStats.to_dict returns correct keys."""
        stats = TimingStats(count=10, mean=0.01, std=0.001, min=0.005, max=0.02, p50=0.01, p90=0.015, p95=0.018, p99=0.02)
        d = stats.to_dict()
        assert d["count"] == 10
        assert d["mean"] == 0.01
        assert d["p50"] == 0.01
        assert d["p99"] == 0.02


class TestComputeTimingStats:
    """Tests for compute_timing_stats."""

    def test_single_value(self):
        """Single value returns zero std."""
        stats = compute_timing_stats([0.01])
        assert stats.count == 1
        assert stats.mean == 0.01
        assert stats.std == pytest.approx(0.0, abs=1e-15)
        assert stats.min == 0.01
        assert stats.max == 0.01
        assert stats.p50 == 0.01

    def test_multiple_values(self):
        """Multiple values compute correct stats."""
        times = [0.01, 0.02, 0.03, 0.04, 0.05]
        stats = compute_timing_stats(times)
        assert stats.count == 5
        assert abs(stats.mean - 0.03) < 1e-10
        assert stats.min == 0.01
        assert stats.max == 0.05

    def test_empty_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_timing_stats([])

    def test_percentiles(self):
        """Percentiles are computed correctly."""
        times = list(range(1, 101))  # 1 to 100
        times_float = [t / 1000.0 for t in times]
        stats = compute_timing_stats(times_float)
        assert stats.p50 == pytest.approx(0.0505, abs=0.001)
        assert stats.count == 100

    def test_all_same(self):
        """All identical values return same percentile."""
        times = [0.01] * 10
        stats = compute_timing_stats(times)
        assert stats.p50 == 0.01
        assert stats.p90 == 0.01
        assert stats.p95 == 0.01
        assert stats.p99 == 0.01
        assert stats.std == pytest.approx(0.0, abs=1e-15)


class TestComputeFPS:
    """Tests for compute_fps."""

    def test_basic(self):
        """FPS = count / total_time."""
        fps = compute_fps([0.01] * 100)
        assert fps == pytest.approx(100.0 / 1.0)

    def test_empty(self):
        """Empty list returns 0."""
        assert compute_fps([]) == 0.0

    def test_zero_total(self):
        """Zero total time returns 0."""
        assert compute_fps([0.0]) == 0.0

    def test_negative_time(self):
        """Negative total time returns 0."""
        assert compute_fps([-0.01]) == 0.0
