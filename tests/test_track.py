from pathlib import Path

import pytest

from vision_platform.detect import FakeDetector, YoloxOnnxDetector
from vision_platform.sources import VideoFileSource
from vision_platform.track import ByteTrackAdapter
from vision_platform.types import Detection

ROOT = Path(__file__).parent.parent


def det(x, y=100.0, score=0.9):
    return Detection(bbox=(x, y, x + 60, y + 160), score=score, class_id=0)


def test_single_moving_box_keeps_one_track_id():
    frames = [[det(100 + i * 8)] for i in range(10)]
    tracker = ByteTrackAdapter(camera_id="cam")
    ids = set()
    for i, dets in enumerate(frames):
        for o in tracker.update(dets, ts=i * 0.2, frame_size=(1920, 1080)):
            ids.add(o.track_id)
    assert len(ids) == 1


def test_two_boxes_get_two_track_ids():
    tracker = ByteTrackAdapter(camera_id="cam")
    ids = set()
    for i in range(5):
        obs = tracker.update([det(100 + i * 5), det(900 - i * 5)], ts=i * 0.2,
                             frame_size=(1920, 1080))
        ids.update(o.track_id for o in obs)
    assert len(ids) == 2


def test_fake_detector_replays_script_then_empty():
    fd = FakeDetector([[det(0)], [det(10), det(50)]])
    import numpy as np
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert len(fd.detect(frame)) == 1
    assert len(fd.detect(frame)) == 2
    assert fd.detect(frame) == []


def test_missing_video_raises_before_yielding(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError), match="nope.mp4"):
        VideoFileSource(tmp_path / "nope.mp4", analytics_fps=5)


def test_missing_detector_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.onnx"):
        YoloxOnnxDetector(tmp_path / "nope.onnx", providers=["CPUExecutionProvider"])


@pytest.mark.model
def test_yolox_finds_people_on_labkom_frame():
    model = ROOT / "models" / "object_detection_yolox_2022nov.onnx"
    sample = Path.home() / "Downloads" / (
        "screenshot_Labkom_2_LABKOM_DAN_KELAS_C13_LABKOM_DAN_KELAS_C13"
        "_20261029070002_1793469586208.png")
    if not model.exists() or not sample.exists():
        pytest.skip("YOLOX weights or Labkom sample absent")
    import cv2
    # This ONNX export consumes BGR-ordered 0..255 input (verified empirically
    # against RGB on this frame: 4x the candidate rows), and the wide shot
    # needs native-resolution tiling: whole-frame 640px inference finds only
    # a handful of the ~30 seated persons.
    detector = YoloxOnnxDetector(model, providers=["CPUExecutionProvider"],
                                 score_threshold=0.35, tile_grid=(2, 2))
    dets = detector.detect(cv2.imread(str(sample)))
    # Seated occlusion makes an exact count the wrong assertion;
    # the detector must find a meaningful subset.
    assert len(dets) >= 10, f"only {len(dets)} persons found"
    assert all(d.class_id == 0 for d in dets)
