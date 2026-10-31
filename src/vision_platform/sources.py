"""Frame sources. POC scope: video files (and, through OpenCV, webcams/RTSP
URLs passed as paths — those are exercised at site, not in tests)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class VideoFileSource:
    """Yields (ts_seconds, frame_bgr), throttled to `analytics_fps` by dropping
    frames. Latency is never accumulated: skipped frames are simply not analyzed."""

    def __init__(self, path: str | Path, analytics_fps: float = 5.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video file not found: {path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        self._cap = cap
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.src_fps = float(src_fps)
        self._step = max(1, round(self.src_fps / max(analytics_fps, 0.1)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (width, height)

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            if idx % self._step == 0:
                yield idx / self.src_fps, frame
            idx += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoFileSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
