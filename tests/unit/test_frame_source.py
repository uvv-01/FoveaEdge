"""Unit tests for foveaedge.benchmark.frame_source — FrameSource."""

from __future__ import annotations

import numpy as np
import pytest

from foveaedge.benchmark.frame_source import SyntheticFrameSource


class TestSyntheticFrameSource:
    """Tests for SyntheticFrameSource."""

    def test_creation(self):
        """Can create a synthetic frame source."""
        src = SyntheticFrameSource(count=10, width=32, height=32, channels=3)
        assert len(src) == 10

    def test_deterministic(self):
        """Same seed produces same frames."""
        src1 = SyntheticFrameSource(count=5, seed=42)
        src2 = SyntheticFrameSource(count=5, seed=42)
        for i in range(5):
            np.testing.assert_array_equal(src1[i], src2[i])

    def test_different_seeds(self):
        """Different seeds produce different frames."""
        src1 = SyntheticFrameSource(count=5, seed=42)
        src2 = SyntheticFrameSource(count=5, seed=99)
        assert not np.array_equal(src1[0], src2[0])

    def test_frame_shape(self):
        """frame_shape returns correct dimensions."""
        src = SyntheticFrameSource(count=5, width=64, height=48, channels=3)
        assert src.frame_shape() == (48, 64, 3)

    def test_grayscale(self):
        """Grayscale frames have shape (H, W)."""
        src = SyntheticFrameSource(count=5, width=32, height=32, channels=1)
        assert src.frame_shape() == (32, 32)
        assert src[0].ndim == 2

    def test_rgb(self):
        """RGB frames have shape (H, W, 3)."""
        src = SyntheticFrameSource(count=5, width=32, height=32, channels=3)
        assert src[0].ndim == 3
        assert src[0].shape[2] == 3

    def test_dtype_uint8(self):
        """Frames are uint8."""
        src = SyntheticFrameSource(count=5)
        assert src[0].dtype == np.uint8

    def test_value_range(self):
        """Frame values are in [0, 255]."""
        src = SyntheticFrameSource(count=50, width=64, height=64)
        for i in range(50):
            assert src[i].min() >= 0
            assert src[i].max() <= 255

    def test_index_error(self):
        """Out-of-range index raises IndexError."""
        src = SyntheticFrameSource(count=5)
        with pytest.raises(IndexError):
            _ = src[5]
        with pytest.raises(IndexError):
            _ = src[-1]

    def test_invalid_count(self):
        """Zero count raises ValueError."""
        with pytest.raises(ValueError):
            SyntheticFrameSource(count=0)

    def test_invalid_channels(self):
        """Invalid channel count raises ValueError."""
        with pytest.raises(ValueError):
            SyntheticFrameSource(channels=2)

    def test_invalid_dimensions(self):
        """Zero dimensions raise ValueError."""
        with pytest.raises(ValueError):
            SyntheticFrameSource(width=0)
        with pytest.raises(ValueError):
            SyntheticFrameSource(height=0)
