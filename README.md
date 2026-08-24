# FoveaEdge

Selective inference research for OpenVINO edge devices.

## Research Question

Can high-resolution edge vision systems avoid running expensive inference over the entire frame on every frame by dynamically selecting only spatially important regions, while maintaining acceptable accuracy and predictable latency?

## Status

| Component | Status |
|-----------|--------|
| Project foundation | Implemented |
| OpenVINO model/inference | Planned |
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
# Run smoke test
python -m pytest tests/ -v
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
├── src/foveaedge/        # Main package
├── tests/                # Test suite
├── docs/                 # Documentation
├── models/               # Model files and generators
├── benchmark/            # Benchmark framework
├── pyproject.toml        # Project configuration
└── README.md
```

## License

MIT
