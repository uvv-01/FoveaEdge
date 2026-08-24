"""Frame source abstraction for benchmark input.

Provides deterministic synthetic frames and extensible video file support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class FrameSource(ABC):
    """Abstract base class for frame sources.

    Subclasses must implement __len__ and __getitem__.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of available frames."""

    @abstractmethod
    def __getitem__(self, index: int) -> np.ndarray:
        """Return a frame at the given index.

        Args:
            index: Frame index (0-based).

        Returns:
            Frame as numpy array (H, W, C) or (H, W).
        """

    @abstractmethod
    def frame_shape(self) -> tuple[int, ...]:
        """Return the shape of a single frame."""


class SyntheticFrameSource(FrameSource):
    """Deterministic synthetic frame source for infrastructure testing.

    Generates frames using a seeded RNG for reproducibility.

    Args:
        count: Number of frames to generate.
        width: Frame width.
        height: Frame height.
        channels: Number of channels (1 or 3).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        count: int = 100,
        width: int = 32,
        height: int = 32,
        channels: int = 3,
        seed: int = 42,
    ) -> None:
        if count <= 0:
            raise ValueError("count must be > 0")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")

        self._count = count
        self._width = width
        self._height = height
        self._channels = channels
        self._seed = seed

        # Pre-generate all frames deterministically
        rng = np.random.RandomState(seed)
        if channels == 1:
            self._frames = rng.randint(0, 255, (count, height, width), dtype=np.uint8)
        else:
            self._frames = rng.randint(0, 255, (count, height, width, channels), dtype=np.uint8)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._count:
            raise IndexError(f"Frame index {index} out of range [0, {self._count})")
        return self._frames[index]

    def frame_shape(self) -> tuple[int, ...]:
        if self._channels == 1:
            return (self._height, self._width)
        return (self._height, self._width, self._channels)

    @property
    def seed(self) -> int:
        """Return the random seed used for frame generation."""
        return self._seed


class VideoFrameSource(FrameSource):
    """Video file frame source using OpenCV.

    Reads frames from a video file. Falls back gracefully if OpenCV
    is not available or the file cannot be opened.

    Args:
        video_path: Path to a video file.
        max_frames: Maximum number of frames to read (0 = all).
    """

    def __init__(self, video_path: str | Path, max_frames: int = 0) -> None:
        self._video_path = Path(video_path)
        if not self._video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self._video_path}")

        try:
            import cv2

            self._cap = cv2.VideoCapture(str(self._video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self._video_path}")

            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames > 0:
                self._frame_count = min(self._frame_count, max_frames)

            # Read first frame to determine shape
            ret, first_frame = self._cap.read()
            if not ret or first_frame is None:
                raise RuntimeError("Cannot read first frame from video")

            # OpenCV returns BGR
            self._frame_shape = first_frame.shape
            self._first_frame = first_frame
            self._frame_index = 0

        except ImportError:
            raise RuntimeError("OpenCV is required for VideoFrameSource")

    def __len__(self) -> int:
        return self._frame_count

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._frame_count:
            raise IndexError(f"Frame index {index} out of range [0, {self._frame_count})")

        # Seek to the requested frame
        self._cap.set(1, index)  # cv2.CAP_PROP_POS_FRAMES
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Cannot read frame {index}")

        # Convert BGR to RGB
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def frame_shape(self) -> tuple[int, ...]:
        return self._frame_shape

    def close(self) -> None:
        """Release the video capture resource."""
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        self.close()
