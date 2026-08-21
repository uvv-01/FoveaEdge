"""Tests for foveaedge.environment."""

import pytest

from foveaedge.environment import detect_environment, EnvironmentInfo, DeviceInfo


class TestDeviceInfo:
    def test_defaults(self):
        dev = DeviceInfo(name="CPU", full_name="Intel(R) Core")
        assert dev.name == "CPU"
        assert dev.full_name == "Intel(R) Core"
        assert dev.supported_metrics == []


class TestEnvironmentInfo:
    def test_to_dict(self):
        info = EnvironmentInfo(
            python_version="3.11.4",
            openvino_version="2024.6.0",
            primary_device="CPU",
        )
        d = info.to_dict()
        assert d["python_version"] == "3.11.4"
        assert d["primary_device"] == "CPU"
        assert d["devices"] == []

    def test_summary(self):
        info = EnvironmentInfo(
            python_version="3.11.4",
            openvino_version="2024.6.0",
            primary_device="CPU",
        )
        summary = info.summary()
        assert "3.11.4" in summary
        assert "2024.6.0" in summary
        assert "CPU" in summary


class TestDetectEnvironment:
    def test_detect(self):
        """Test that environment detection works with real OpenVINO."""
        info = detect_environment()
        assert isinstance(info, EnvironmentInfo)
        assert info.python_version != ""
        assert info.openvino_version != ""
        assert len(info.devices) > 0
        assert info.primary_device in [d.name for d in info.devices]

    def test_has_cpu(self):
        """CPU should always be available."""
        info = detect_environment()
        device_names = [d.name for d in info.devices]
        assert "CPU" in device_names

    def test_device_details(self):
        """Each device should have name and full_name."""
        info = detect_environment()
        for dev in info.devices:
            assert dev.name != ""
            assert dev.full_name != ""
