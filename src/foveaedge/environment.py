"""Environment information for reproducibility.

Captures hardware and software environment details that affect
benchmark results and reproducibility.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field

try:
    import openvino

    OPENVINO_VERSION = openvino.__version__
except ImportError:
    OPENVINO_VERSION = "not installed"

try:
    import numpy

    NUMPY_VERSION = numpy.__version__
except ImportError:
    NUMPY_VERSION = "not installed"

try:
    import cv2

    OPENCV_VERSION = cv2.__version__
except ImportError:
    OPENCV_VERSION = "not installed"


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Snapshot of the execution environment for reproducibility.

    Attributes:
        python_version: Python version string.
        os_name: Operating system name.
        os_release: OS release version.
        os_machine: Machine architecture.
        hostname: Machine hostname.
        openvino_version: OpenVINO version.
        numpy_version: NumPy version.
        opencv_version: OpenCV version.
        available_devices: List of available OpenVINO devices.
        cpu_count: Number of CPU cores.
    """

    python_version: str = field(default_factory=lambda: sys.version)
    os_name: str = field(default_factory=lambda: platform.system())
    os_release: str = field(default_factory=lambda: platform.release())
    os_machine: str = field(default_factory=lambda: platform.machine())
    hostname: str = field(default_factory=lambda: platform.node())
    openvino_version: str = OPENVINO_VERSION
    numpy_version: str = NUMPY_VERSION
    opencv_version: str = OPENCV_VERSION
    available_devices: tuple[str, ...] = ()
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "python_version": self.python_version,
            "os_name": self.os_name,
            "os_release": self.os_release,
            "os_machine": self.os_machine,
            "hostname": self.hostname,
            "openvino_version": self.openvino_version,
            "numpy_version": self.numpy_version,
            "opencv_version": self.opencv_version,
            "available_devices": list(self.available_devices),
            "cpu_count": self.cpu_count,
        }


def capture_environment() -> EnvironmentSnapshot:
    """Capture the current environment as a snapshot.

    Returns:
        EnvironmentSnapshot with current system information.
    """
    try:
        from openvino.runtime import Core

        core = Core()
        devices = tuple(core.available_devices)
    except ImportError:
        devices = ()

    return EnvironmentSnapshot(available_devices=devices)
