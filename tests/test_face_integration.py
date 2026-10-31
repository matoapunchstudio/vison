"""Integration tests against real YuNet+SFace weights. Skipped (with reason)
when weights are absent; see models/README.md for fetch commands."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
DET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
REC = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
SAMPLE = Path.home() / "Downloads" / (
    "screenshot_Labkom_2_LABKOM_DAN_KELAS_C13_LABKOM_DAN_KELAS_C13"
    "_20261029070002_1793469586208.png")

pytestmark = pytest.mark.model


def _encoder():
    if not (DET.exists() and REC.exists()):
        pytest.skip("face weights absent in models/ (see models/README.md)")
    from vision_platform.face import YuNetSFaceEncoder
    return YuNetSFaceEncoder(DET, REC)


def test_blank_frame_yields_no_faces():
    assert _encoder().embed(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_sample_frame_faces_and_determinism():
    if not SAMPLE.exists():
        pytest.skip("Labkom sample frame not present on this machine")
    import cv2
    enc = _encoder()
    frame = cv2.imread(str(SAMPLE))
    hits1 = enc.embed(frame)
    # Measured 2026-11-01 on this frame: 2 faces pass score>=0.6 and width>=40px
    # out of ~30 seated students — the steep, wide shot is count-friendly but
    # face-hostile, exactly as the research report predicts. Assert only that
    # the encoder works end to end on this camera's geometry.
    assert len(hits1) >= 1, f"expected at least one usable face, got {len(hits1)}"
    for h in hits1:
        assert h.embedding.shape == (128,)
        assert abs(float(np.linalg.norm(h.embedding)) - 1.0) < 1e-4
    hits2 = enc.embed(frame)
    same = float(np.dot(hits1[0].embedding, hits2[0].embedding))
    assert same > 0.99  # deterministic: same face -> same vector
