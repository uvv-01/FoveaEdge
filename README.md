# FoveaEdge

Selective inference research for OpenVINO edge devices.

## Research Question

Can high-resolution edge vision systems avoid running expensive inference over the entire frame on every frame by dynamically selecting only spatially important regions, while maintaining acceptable accuracy and predictable latency?

## Status

| Component | Status |
|-----------|--------|
| Project foundation | Implemented |
| OpenVINO model/inference | Implemented |
| Reproducible baseline | **Implemented** |
| Region representation | **Implemented** |
| ROI extraction | Planned |
| ROI inference | Planned |
| Baseline vs ROI benchmark | Planned |
| Spatial selector | Planned |
| Temporal stability | Planned |
| Adaptive ROI policy | Planned |
| Accuracy evaluation | Planned |
| Compute-aware scheduler | Planned |
| Hardware-aware policy | Planned |
| Async inference | Planned |
| Event-driven pipeline | Planned |
| Stress testing | Planned |
| Experiment matrix | Planned |
| Research analysis | Planned |
| OpenVINO integration audit | Planned |
| Reproducibility package | Planned |

## Current Implementation (Day 4)

### OpenVINO Inference (Day 2)
- **ModelLoader**: Load and compile OpenVINO models with device validation
- **ModelInfo / TensorInfo**: Structured metadata for model inputs/outputs
- **InferenceEngine**: Synchronous inference with per-stage timing (preprocessing, inference, postprocessing)
- **Preprocessing**: Resize, normalization, HWC->NCHW conversion
- **Test model**: Deterministic Conv->ReLU->GAP->FC model (10 classes, 32x32 input)

### Benchmark Baseline (Day 3)
- **BenchmarkConfig**: Reproducible experiment configuration (warmup, measured frames, seed, device)
- **BenchmarkResult**: Structured results with JSON output and human-readable summary
- **BaselineRunner**: Full-frame inference benchmark with warmup exclusion
- **TimingStats**: P50, P90, P95, P99, FPS, mean, std, min, max
- **FrameSource**: Deterministic synthetic frames + extensible video file support
- **EnvironmentSnapshot**: Software/hardware environment capture for reproducibility

### Region Representation (Day 4)
- **Region**: Rectangular region with validation, clipping, intersection detection
- **RegionSet**: Collection with add/remove, bounding box, area, filtering
- **Geometry**: intersection_area, IoU, overlap_ratio, contains_point, contains_region, clip_region

### Not yet implemented
- ROI extraction and ROI inference
- Spatial selection / motion detection
- Temporal tracking
- Scheduling
- Async inference
- Benchmark comparison (full-frame vs selective)

## Installation

```bash
# Clone the repository
git clone https://github.com/uvv-01/FoveaEdge.git
cd FoveaEdge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run all tests (189 tests)
python -m pytest tests/ -v

# Generate test model
python models/generate_test_model.py models/test_model

# Check OpenVINO status
python -m foveaedge status
```

## Usage Example

```python
from foveaedge.model import ModelLoader
from foveaedge.inference.engine import InferenceEngine
from foveaedge.benchmark.config import BenchmarkConfig
from foveaedge.benchmark.runner import BaselineRunner
from foveaedge.regions import Region, RegionSet
import numpy as np

# Load and compile a model
loader = ModelLoader()
loader.load_and_compile("models/test_model/test_model.xml", device="CPU")

# Run single inference
engine = InferenceEngine(loader)
image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
result = engine.infer(image)
print(f"Inference time: {result.inference_time_s:.4f}s")

# Run full benchmark
config = BenchmarkConfig(
    model_path="models/test_model/test_model.xml",
    warmup_frames=10,
    measured_frames=100,
)
runner = BaselineRunner(config)
bench_result = runner.run()
print(bench_result.summary())
# Save as JSON
with open("benchmark_result.json", "w") as f:
    f.write(bench_result.to_json())

# Work with regions
r1 = Region(x=10, y=10, width=50, height=50)
r2 = Region(x=30, y=30, width=50, height=50)
rs = RegionSet([r1, r2])
print(f"Total area: {rs.total_area}")
print(f"Bounding box: {rs.bounding_box}")
```

## Architecture

```
Frame Source
      ↓
Temporal State
      ↓
Spatial Selector
      ↓
Scheduler
      ↓
 ┌───────────────┬───────────────┐
 ↓                               ↓
Full Frame                    ROI Path
 ↓                               ↓
Inference                    ROI Extraction
 ↓                               ↓
                          ROI Inference
 └───────────────┬───────────────┘
                 ↓
            Result Mapping
                 ↓
             Telemetry
                 ↓
           Benchmarking
```

Day 3 establishes the **Full Frame** branch (baseline).
Day 4 establishes the **Region/ROI primitive** required by the future ROI branch.

## Project Structure

```
FoveaEdge/
├── src/foveaedge/              # Main package
│   ├── __init__.py             # Package version
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # Configuration dataclasses
│   ├── environment.py          # Environment snapshot
│   ├── model.py                # ModelLoader, ModelInfo, TensorInfo
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py           # InferenceEngine with timing
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── config.py           # BenchmarkConfig
│   │   ├── frame_source.py     # Synthetic + Video frame sources
│   │   ├── result.py           # BenchmarkResult with JSON
│   │   ├── runner.py           # BaselineRunner
│   │   └── stats.py            # TimingStats, percentiles, FPS
│   └── regions/
│       ├── __init__.py
│       ├── region.py           # Region with validation/clipping
│       ├── region_set.py       # RegionSet collection
│       └── geometry.py         # IoU, intersection, containment
├── tests/
│   ├── unit/                   # 159 unit tests
│   │   ├── test_model.py
│   │   ├── test_inference.py
│   │   ├── test_benchmark_config.py
│   │   ├── test_benchmark_stats.py
│   │   ├── test_benchmark_result.py
│   │   ├── test_frame_source.py
│   │   ├── test_region.py
│   │   ├── test_region_set.py
│   │   └── test_geometry.py
│   ├── integration/            # 17 integration tests
│   │   ├── test_inference_pipeline.py
│   │   └── test_baseline_runner.py
│   └── test_smoke.py           # 10 smoke tests
├── models/
│   ├── generate_test_model.py
│   └── test_model/             # Generated test model files
├── docs/
├── pyproject.toml
└── README.md
```

## Benchmark Methodology

The baseline benchmark:

1. **Warmup**: Runs N inference passes (default: 10) excluded from timing
2. **Measurement**: Runs M inference passes (default: 100) with per-stage timing
3. **Statistics**: Computes P50, P90, P95, P99, FPS, mean, std for each stage
4. **Reproducibility**: Captures environment, config, seed, and device info
5. **Output**: Machine-readable JSON + human-readable summary

The baseline uses **deterministic synthetic frames** for infrastructure testing.
This is NOT a research accuracy benchmark — it measures inference cost only.

## Dependencies

- Python >= 3.10
- OpenVINO >= 2024.0
- NumPy >= 1.24, < 2.0
- OpenCV >= 4.8
- ONNX / ONNXRuntime (for test model generation)

## License

MIT
