# Day 01 — Repository + Architecture Foundation

## Objective

Initialize the FoveaEdge project with proper Python packaging, configuration system,
environment detection, and test infrastructure.

## Work Completed

- Created `.gitignore` for Python/OpenVINO projects
- Created `pyproject.toml` with project metadata and dependencies
- Scaffolded `src/foveaedge/` package with all subsystem subpackages:
  - `capture`, `peripheral`, `roi`, `scheduler`, `inference`, `fusion`, `tracking`, `telemetry`, `ui`
- Created `foveaedge.config` — dataclass-based configuration with YAML/JSON serialization
- Created `foveaedge.environment` — OpenVINO device discovery and environment reporting
- Created `foveaedge.cli` — command-line entry points (info, bench)
- Created `default_config.yaml` — default configuration for all subsystems
- Created test infrastructure: `conftest.py`, `tests/unit/`, `tests/integration/`
- Created `benchmark/__init__.py`
- Created `docs/progress/`, `docs/architecture/`, `docs/methodology/`

## Files Changed

- `.gitignore` (new)
- `pyproject.toml` (new)
- `default_config.yaml` (new)
- `src/foveaedge/__init__.py` (new)
- `src/foveaedge/config.py` (new)
- `src/foveaedge/environment.py` (new)
- `src/foveaedge/cli.py` (new)
- `src/foveaedge/{capture,peripheral,roi,scheduler,inference,fusion,tracking,telemetry,ui}/__init__.py` (new)
- `tests/__init__.py` (new)
- `tests/conftest.py` (new)
- `tests/unit/__init__.py` (new)
- `tests/unit/test_config.py` (new)
- `tests/unit/test_environment.py` (new)
- `tests/integration/__init__.py` (new)
- `benchmark/__init__.py` (new)
- `docs/progress/day-01.md` (new)

## Environment

- Python 3.11.4
- OpenVINO 2024.6.0
- OpenCV 4.13.0
- NumPy 1.26.4
- Devices: CPU, GPU

## Problems

- None encountered during foundation setup.

## Decisions

- Used `src/foveaedge/` layout (src-layout) for clean namespace separation.
- Used dataclasses for configuration (no external dependency needed).
- Used YAML as primary config format (PyYAML already installed).
- All scoring weights in SchedulerConfig sum to 1.0 by default.

## Next Step

Day 02: OpenVINO model loading and basic inference smoke test.
