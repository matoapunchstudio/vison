"""Face encoding: detect + 5-point alignment + embedding, as one node.

Production implementation is OpenCV Zoo YuNet (detector, MIT) + SFace
(recognizer, Apache-2.0) — the commercial-clean baseline from the research
report. Encoders sit behind the FaceEncoder protocol so a licensed/self-trained
ArcFace/AdaFace encoder can replace SFace by config later; gallery `model_id`
guards keep embedding spaces from being mixed.

Alignment and preprocessing live inside the encoder, never in pipeline
config: a wrong warp or normalization silently destroys accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vision_platform.types import FaceHit

SFACE_MODEL_ID = "yunet+sface-2021dec/1"


class FaceEncoder(Protocol):
    model_id: str

    def embed(self, frame_bgr: np.ndarray) -> list[FaceHit]: ...


class YuNetSFaceEncoder:
    """YuNet detection + SFace 128-d embeddings (L2-normalized)."""

    model_id = SFACE_MODEL_ID

    def __init__(self, detector_path: str | Path, recognizer_path: str | Path,
                 min_face_px: int = 40, score_threshold: float = 0.6):
        for p in (detector_path, recognizer_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"face model file not found: {p}")
        self.min_face_px = int(min_face_px)
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), score_threshold)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def embed(self, frame_bgr: np.ndarray) -> list[FaceHit]:
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)
        hits: list[FaceHit] = []
        if faces is None:
            return hits
        for face in faces:
            x, y, fw, fh = (float(face[i]) for i in range(4))
            if fw < self.min_face_px:
                continue  # too small to identify reliably; skip early, save compute
            aligned = self._recognizer.alignCrop(frame_bgr, face)
            emb = self._recognizer.feature(aligned).ravel().astype(np.float32)
            norm = float(np.linalg.norm(emb))
            if norm == 0.0:
                continue
            hits.append(FaceHit(bbox=(x, y, x + fw, y + fh),
                                embedding=emb / norm,
                                quality=float(face[14])))
        return hits


class FakeEncoder:
    """Scripted encoder for tests: each embed() returns the next scripted hit list."""

    def __init__(self, script: list[list[FaceHit]] | None = None, model_id: str = "fake/1"):
        self.model_id = model_id
        self._script = list(script or [])
        self.calls = 0

    def embed(self, frame_bgr: np.ndarray) -> list[FaceHit]:
        self.calls += 1
        if self.calls <= len(self._script):
            return self._script[self.calls - 1]
        return []
