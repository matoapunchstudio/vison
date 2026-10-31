"""Core data types shared by every plane.

Frame plane produces Detection -> TrackObservation (+ optional FaceHit).
Event plane consumes TrackObservation and produces MachineEvent.
Bboxes are (x1, y1, x2, y2) in pixel coordinates of the source frame.
All timestamps are float seconds from the source clock (never wall time),
so the event plane is deterministic and unit-testable with synthetic feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    class_id: int


@dataclass(frozen=True)
class TrackObservation:
    camera_id: str
    ts: float
    track_id: int
    bbox: BBox
    frame_size: tuple[int, int]  # (width, height) pixels, for zone normalization
    class_id: int
    identity: str | None = None
    identity_score: float | None = None


@dataclass(frozen=True)
class FaceHit:
    """One detected face and its L2-normalized embedding."""

    bbox: BBox
    embedding: np.ndarray  # float32, unit norm
    quality: float  # detector score in [0, 1]


@dataclass(frozen=True)
class MachineEvent:
    """A composed event ready for payload emission. `data` is event-type specific."""

    event_id: str
    type: str
    ts: float
    camera_id: str
    track_id: int
    identity: str | None
    identity_score: float | None
    data: dict[str, Any] = field(default_factory=dict)
