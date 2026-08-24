"""Smoke tests for FoveaEdge Day 1 — project foundation."""

import importlib


def test_package_imports():
    """Verify foveaedge package is importable."""
    import foveaedge

    assert hasattr(foveaedge, "__version__")
    assert foveaedge.__version__ == "0.1.0"


def test_version_exists():
    """Verify version string is defined."""
    from foveaedge import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_cli_module_importable():
    """Verify CLI module is importable."""
    from foveaedge.cli import main

    assert callable(main)


def test_cli_status():
    """Verify CLI status command works."""
    from foveaedge.cli import main

    result = main(["status"])
    assert result == 0


def test_cli_version():
    """Verify CLI version flag works."""
    from foveaedge.cli import main

    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0


def test_openvino_import():
    """Verify OpenVINO can be imported."""
    try:
        import openvino

        assert hasattr(openvino, "__version__") or hasattr(openvino.runtime, "Core")
    except ImportError:
        import pytest

        pytest.skip("OpenVINO not installed")


def test_numpy_import():
    """Verify NumPy can be imported."""
    import numpy as np

    assert np.__version__


def test_opencv_import():
    """Verify OpenCV can be imported."""
    try:
        import cv2

        assert cv2.__version__
    except ImportError:
        import pytest

        pytest.skip("OpenCV not installed")


def test_openvino_core():
    """Verify OpenVINO Core can be instantiated."""
    try:
        from openvino.runtime import Core

        core = Core()
        devices = core.available_devices
        assert isinstance(devices, list)
        assert len(devices) > 0
    except ImportError:
        import pytest

        pytest.skip("OpenVINO not installed")


def test_package_importlib():
    """Verify package is properly installed and importable via importlib."""
    mod = importlib.import_module("foveaedge")
    assert mod is not None
