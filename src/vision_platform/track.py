"""Tracking adapter: ByteTrack (via the `trackers` package, where Roboflow's
tracking algorithms now live; supervision 0.31 no longer ships ByteTrack)
behind our observation types."""

from __future__ import annotations

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from vision_platform.types import Detection, TrackObservation


class ByteTrackAdapter:
    """Assigns persistent track_ids to detections (motion-based, no appearance model)."""

    def __init__(self, camera_id: str, frame_rate: int = 5):
        self.camera_id = camera_id
        self._tracker = ByteTrackTracker(frame_rate=float(frame_rate))

    def update(self, detections: list[Detection], ts: float,
               frame_size: tuple[int, int]) -> list[TrackObservation]:
        if detections:
            sv_dets = sv.Detections(
                xyxy=np.array([d.bbox for d in detections], dtype=np.float32),
                confidence=np.array([d.score for d in detections], dtype=np.float32),
                class_id=np.array([d.class_id for d in detections]),
            )
        else:
            sv_dets = sv.Detections.empty()
        tracked = self._tracker.update(sv_dets)
        out: list[TrackObservation] = []
        for xyxy, _mask, _conf, class_id, tracker_id, _data in tracked:
            if tracker_id is None or int(tracker_id) < 0:
                continue  # -1 = detection not yet confirmed as a track
            out.append(TrackObservation(
                camera_id=self.camera_id, ts=ts, track_id=int(tracker_id),
                bbox=tuple(float(v) for v in xyxy), frame_size=frame_size,
                class_id=int(class_id) if class_id is not None else 0))
        return out
