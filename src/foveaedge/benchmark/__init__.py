"""Benchmark framework for FoveaEdge selective inference experiments.

Provides reproducible full-frame baseline measurement.
"""

from foveaedge.benchmark.config import BenchmarkConfig
from foveaedge.benchmark.runner import BaselineRunner
from foveaedge.benchmark.result import BenchmarkResult

__all__ = ["BenchmarkConfig", "BenchmarkResult", "BaselineRunner"]
