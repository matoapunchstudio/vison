"""Object/person detection behind a small protocol.

Production implementation: YOLOX (OpenCV Zoo ONNX, Apache-2.0) on ONNX
Runtime. Pre/post-processing follows the OpenCV Zoo reference implementation
(models/object_detection_yolox): letterbox with 114 padding, no 0-1 scaling,
anchor-grid decode, batched NMS — except channel order: this ONNX export
empirically expects BGR-ordered input (on the Labkom frame, BGR produced 28
candidate rows vs 9 for RGB and 4x tiled detections), so frames are fed as
OpenCV delivers them. Wide shots can be tiled (`tile_grid`) so small persons
are inferred at native resolution instead of being shrunk into 640 px.
Execution providers come from the runtime profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

from vision_platform.types import Detection

_STRIDES = (8, 16, 32)
_INPUT = 640


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def _grids() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in _STRIDES:
        hs = ws = _INPUT // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides.append(np.full((1, grid.shape[1], 1), s))
    return np.concatenate(grids, 1), np.concatenate(strides, 1)


class YoloxOnnxDetector:
    """YOLOX person/object detector; returns detections in source-frame pixels."""

    def __init__(self, model_path: str | Path, providers: list[str],
                 score_threshold: float = 0.4, nms_threshold: float = 0.5,
                 class_ids: tuple[int, ...] = (0,),
                 tile_grid: tuple[int, int] = (1, 1), tile_overlap: float = 0.12):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"detector model file not found: {model_path}")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.class_ids = class_ids  # COCO ids to keep; 0 = person
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.active_providers = self._session.get_providers()
        self._grids, self._strides = _grids()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        gx, gy = self.tile_grid
        if (gx, gy) == (1, 1):
            return self._detect_single(frame_bgr)
        h, w = frame_bgr.shape[:2]
        tw, th = int(np.ceil(w / gx)), int(np.ceil(h / gy))
        ox, oy = int(tw * self.tile_overlap), int(th * self.tile_overlap)
        merged: list[Detection] = []
        for row in range(gy):
            for col in range(gx):
                x0 = max(0, col * tw - (ox if col else 0))
                y0 = max(0, row * th - (oy if row else 0))
                x1 = min(w, (col + 1) * tw + (ox if col < gx - 1 else 0))
                y1 = min(h, (row + 1) * th + (oy if row < gy - 1 else 0))
                for d in self._detect_single(frame_bgr[y0:y1, x0:x1]):
                    bx1, by1, bx2, by2 = d.bbox
                    merged.append(Detection(
                        bbox=(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                        score=d.score, class_id=d.class_id))
        if not merged:
            return []
        boxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]]
                 for d in merged]
        keep = cv2.dnn.NMSBoxes(boxes, [d.score for d in merged],
                                self.score_threshold, self.nms_threshold)
        return [merged[int(i)] for i in np.asarray(keep).ravel()]

    def _detect_single(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        ratio = min(_INPUT / h, _INPUT / w)
        resized = cv2.resize(frame_bgr,
                             (int(w * ratio), int(h * ratio))).astype(np.float32)
        padded = np.full((_INPUT, _INPUT, 3), 114.0, dtype=np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = np.transpose(padded, (2, 0, 1))[np.newaxis]
        out = self._session.run(None, {self._input_name: blob})[0][0].copy()

        out[:, :2] = (out[:, :2] + self._grids[0]) * self._strides[0]
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides[0]
        boxes_xywh = np.empty_like(out[:, :4])
        boxes_xywh[:, 0] = out[:, 0] - out[:, 2] / 2
        boxes_xywh[:, 1] = out[:, 1] - out[:, 3] / 2
        boxes_xywh[:, 2:] = out[:, 2:4]
        scores = out[:, 4:5] * out[:, 5:]
        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)
        keep = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(), max_scores.tolist(), class_ids.tolist(),
            self.score_threshold, self.nms_threshold)
        detections: list[Detection] = []
        for i in np.asarray(keep).ravel():
            cls = int(class_ids[i])
            if cls not in self.class_ids:
                continue
            x, y, bw, bh = (float(v) / ratio for v in boxes_xywh[i])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
            if x2 > x1 and y2 > y1:
                detections.append(Detection(bbox=(x1, y1, x2, y2),
                                            score=float(max_scores[i]), class_id=cls))
        return detections


class FakeDetector:
    """Scripted detector for tests: replays detection lists, then returns empty."""

    def __init__(self, script: list[list[Detection]]):
        self._script = list(script)
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        self.calls += 1
        return self._script[self.calls - 1] if self.calls <= len(self._script) else []
