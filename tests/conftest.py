"""Shared test fixtures for FoveaEdge."""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for imports
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def config():
    """Provide a default Config instance."""
    from foveaedge.config import Config
    return Config()
