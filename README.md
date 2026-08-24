# FoveaEdge

Selective inference research for OpenVINO edge devices.

## Research Question

Can high-resolution edge vision systems avoid running expensive inference over the entire frame on every frame by dynamically selecting only spatially important regions, while maintaining acceptable accuracy and predictable latency?

## Status

| Component | Status |
|-----------|--------|
| Project foundation | Implemented |
| OpenVINO model/inference | **Implemented** |
| Reproducible baseline | Planned |
| Region representation | Planned |
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

## Current Implementation (Day 2)

- **ModelLoader**: Load and compile OpenVINO models with device validation
- **ModelInfo / TensorInfo**: Structured metadata for model inputs/outputs
- **InferenceEngine**: Synchronous inference with per-stage timing (preprocessing, inference, postprocessing)
- **Preprocessing**: Resize, normalization, HWC->NCHW conversion
- **Test model**: Deterministic Conv->ReLU->GAP->FC model (10 classes, 32x32 input)
- **70 tests**: Unit tests for model loading, tensor info, inference engine; integration tests for full pipeline

### Not yet implemented

- Selective ROI inference
- Spatial selection / motion detection
- Temporal tracking
- Scheduling
- Async inference
- Benchmark comparison phase

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
# Run all tests
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
import numpy as np

# Load and compile a model
loader = ModelLoader()
loader.load_and_compile("models/test_model/test_model.xml", device="CPU")

# Create inference engine
engine = InferenceEngine(loader)

# Run inference
image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
result = engine.infer(image)

print(f"Output shape: {result.first_output.shape}")
print(f"Inference time: {result.inference_time_s:.4f}s")
print(f"Total time: {result.total_time_s:.4f}s")
```

## Architecture

```
Frame Source
      |
      v
Temporal State
      |
      v
Spatial Selector
      |
      v
Scheduler
      |
      +----------------------+
      |                      |
      v                      v
Full Frame              ROI Extraction
      |                      |
      |                      v
      |                ROI Inference
      |                      |
      +----------+-----------+
                 |
                 v
           Result Mapping
                 |
                 v
             Telemetry
                 |
                 v
          Benchmark Results
```

## Project Structure

```
FoveaEdge/
├── src/foveaedge/              # Main package
│   ├── __init__.py             # Package version
│   ├── cli.py                  # CLI entry point
│   ├── config.py               # Configuration dataclasses
│   ├── environment.py          # Environment snapshot for reproducibility
│   ├── model.py                # ModelLoader, ModelInfo, TensorInfo
│   └── inference/
│       ├── __init__.py         # Inference package
│       └── engine.py           # InferenceEngine with timing
├── tests/
│   ├── unit/                   # Unit tests
│   │   ├── test_model.py       # ModelLoader tests (18 tests)
│   │   └── test_inference.py   # InferenceEngine tests (23 tests)
│   ├── integration/            # Integration tests
│   │   └── test_inference_pipeline.py  # Full pipeline tests (8 tests)
│   └── test_smoke.py           # Day 1 smoke tests (10 tests)
├── models/
│   ├── generate_test_model.py  # Test model generator
│   └── test_model/             # Generated model files (.xml, .bin, .onnx)
├── docs/                       # Documentation
├── benchmark/                  # Benchmark framework (planned)
├── pyproject.toml              # Project configuration
└── README.md
```

## Test Model

The project includes a minimal deterministic test model for infrastructure testing:

```
Input (1, 3, 32, 32)
  -> Conv2d(3, 16, 3x3, padding=1)
  -> ReLU
  -> Global Average Pooling
  -> Fully Connected(16, 10)
  -> Output (1, 10)
```

This model exists only to test the loading, compilation, and inference infrastructure. It is NOT a research model and does not perform meaningful detection or classification.

## Dependencies

- Python >= 3.10
- OpenVINO >= 2024.0
- NumPy >= 1.24, < 2.0
- OpenCV >= 4.8
- ONNX / ONNXRuntime (for test model generation)

## License

MIT
