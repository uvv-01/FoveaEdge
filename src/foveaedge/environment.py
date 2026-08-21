"""Environment detection and device discovery for FoveaEdge.

Detects available OpenVINO devices, records hardware information,
and provides an environment report for reproducible benchmarks.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceInfo:
    """Information about a single OpenVINO device."""
    name: str
    full_name: str
    supported_metrics: list[str] = field(default_factory=list)
    supported_config_keys: list[str] = field(default_factory=list)


@dataclass
class EnvironmentInfo:
    """Complete environment snapshot for reproducible benchmarks."""
    python_version: str = ""
    openvino_version: str = ""
    opencv_version: str = ""
    numpy_version: str = ""
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""
    devices: list[DeviceInfo] = field(default_factory=list)
    primary_device: str = "CPU"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "python_version": self.python_version,
            "openvino_version": self.openvino_version,
            "opencv_version": self.opencv_version,
            "numpy_version": self.numpy_version,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "platform_machine": self.platform_machine,
            "devices": [
                {
                    "name": d.name,
                    "full_name": d.full_name,
                    "supported_metrics": d.supported_metrics,
                    "supported_config_keys": d.supported_config_keys,
                }
                for d in self.devices
            ],
            "primary_device": self.primary_device,
        }

    def summary(self) -> str:
        """Human-readable environment summary."""
        lines = [
            "=" * 60,
            "FoveaEdge Environment Report",
            "=" * 60,
            f"  Python:          {self.python_version}",
            f"  OpenVINO:        {self.openvino_version}",
            f"  OpenCV:          {self.opencv_version}",
            f"  NumPy:           {self.numpy_version}",
            f"  Platform:        {self.platform_system} {self.platform_release}",
            f"  Architecture:    {self.platform_machine}",
            f"  Primary device:  {self.primary_device}",
            "",
            "  Available OpenVINO devices:",
        ]
        for dev in self.devices:
            lines.append(f"    - {dev.name} ({dev.full_name})")
            if dev.supported_metrics:
                lines.append(f"      metrics: {', '.join(dev.supported_metrics[:5])}")
        lines.append("=" * 60)
        return chr(10).join(lines)


def detect_environment() -> EnvironmentInfo:
    """Detect the full runtime environment including OpenVINO devices.

    Returns an EnvironmentInfo dataclass with all detected information.
    Raises ImportError if OpenVINO is not installed.
    """
    import openvino
    import cv2
    import numpy

    info = EnvironmentInfo()
    info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info.openvino_version = openvino.__version__
    info.opencv_version = cv2.__version__
    info.numpy_version = numpy.__version__
    info.platform_system = platform.system()
    info.platform_release = platform.release()
    info.platform_machine = platform.machine()

    # Enumerate OpenVINO devices
    core = openvino.Core()
    for device_name in core.available_devices:
        dev = DeviceInfo(
            name=device_name,
            full_name=core.get_property(device_name, "FULL_DEVICE_NAME") if device_name != "AUTO" else "AUTO",
        )
        try:
            dev.supported_metrics = list(core.get_metric(device_name, "SUPPORTED_METRICS"))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            dev.supported_config_keys = list(core.get_metric(device_name, "SUPPORTED_CONFIG_KEYS"))  # type: ignore[arg-type]
        except Exception:
            pass
        info.devices.append(dev)

    # Prefer GPU > CPU as primary device
    device_names = [d.name for d in info.devices]
    if "GPU" in device_names:
        info.primary_device = "GPU"
    elif "CPU" in device_names:
        info.primary_device = "CPU"

    return info


def print_environment() -> None:
    """Detect and print the environment report."""
    info = detect_environment()
    print(info.summary())


if __name__ == "__main__":
    print_environment()
