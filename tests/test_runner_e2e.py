import json
from pathlib import Path

import numpy as np

from vision_platform.detect import FakeDetector
from vision_platform.emit import FileEmitter, Outbox
from vision_platform.face import FakeEncoder
from vision_platform.gallery import Gallery
from vision_platform.pipeline import REGISTRY, PipelineRunner, load_pipeline, parse_pipeline
from vision_platform.types import Detection, FaceHit

ROOT = Path(__file__).parent.parent
FRAME = np.zeros((1000, 1000, 3), dtype=np.uint8)


def person_at(nx, ny=0.9):
    x, y = nx * 1000, ny * 1000
    return Detection(bbox=(x - 30, y - 160, x + 30, y), score=0.9, class_id=0)


def face_inside(nx, emb_seed=0, quality=0.95, width=50.0):
    emb = np.zeros(128, dtype=np.float32)
    emb[emb_seed] = 1.0
    x, y = nx * 1000, 900.0
    return FaceHit(bbox=(x - width / 2, y - 150, x + width / 2, y - 150 + width),
                   embedding=emb, quality=quality)


def spec():
    return parse_pipeline({
        "name": "e2e", "camera_id": "cam",
        "source": {"analytics_fps": 5},
        "runtime": {"profile": "dev-cpu", "providers": ["CPUExecutionProvider"]},
        "zones": {"door": [[0, 0], [0.2, 0], [0.2, 1], [0, 1]],
                  "room": [[0.2, 0], [1, 0], [1, 1], [0.2, 1]]},
        "nodes": [
            {"name": "src", "type": "source.file", "config": {}},
            {"name": "det", "type": "detect.persons", "config": {}},
            {"name": "trk", "type": "track.bytetrack", "config": {}},
            {"name": "face", "type": "face.encode", "config": {"min_face_px": 40}},
            {"name": "match", "type": "face.match", "config": {"match_threshold": 0.5}},
            {"name": "seq", "type": "zone.sequence",
             "config": {"order": ["door", "room"], "within_s": 60, "event_type": "person_entered"}},
            {"name": "deb", "type": "debounce", "config": {"cooldown_s": 300}},
            {"name": "out", "type": "emit.file", "config": {}},
        ],
    })


def run_pipeline(tmp_path, face_mode, gallery=None):
    # Gradual walk door -> room so ByteTrack can associate across frames:
    # 0.08, 0.11, ... in 0.03 steps (30px on a 60px-wide box keeps IoU > 0).
    xs = [round(0.08 + 0.03 * i, 2) for i in range(20)]  # ends at 0.65 (room)
    frames = [(float(i), FRAME) for i in range(len(xs))]
    dets = [[person_at(x)] for x in xs]
    if face_mode == "alice":
        faces = [[face_inside(x)] for x in xs]
    elif face_mode == "small":
        faces = [[face_inside(x, width=20.0)] for x in xs]
    elif face_mode == "none":
        faces = [[] for _ in xs]
    elif face_mode == "late":  # face only becomes visible well after entry
        faces = [[] if i < 14 else [face_inside(x)] for i, x in enumerate(xs)]
    else:
        raise AssertionError(face_mode)
    out = tmp_path / "events.jsonl"
    emitter = FileEmitter(Outbox(tmp_path / "o.db"), out)
    runner = PipelineRunner(spec(), detector=FakeDetector(dets),
                            encoder=FakeEncoder(faces, model_id="fake/1"),
                            gallery=gallery, emitter=emitter, machine_id="test-box")
    stats = runner.run(frames)
    payloads = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    return stats, payloads


def gallery_with_alice():
    emb = np.zeros(128, dtype=np.float32)
    emb[0] = 1.0
    g = Gallery(model_id="fake/1")
    g.enroll("alice", emb)
    return g


def test_entry_with_recognized_face(tmp_path):
    stats, payloads = run_pipeline(tmp_path, "alice", gallery_with_alice())
    assert stats.events == 1 and len(payloads) == 1
    assert payloads[0]["type"] == "person_entered"
    assert payloads[0]["identity"] == "alice"


def test_entry_without_face_is_unresolved_with_reason(tmp_path):
    stats, payloads = run_pipeline(tmp_path, "none", gallery_with_alice())
    assert len(payloads) == 1
    assert payloads[0]["identity"] is None
    assert payloads[0]["data"]["unresolved_reason"] == "no_face_detected"


def test_late_identity_resolution_links_to_entry(tmp_path):
    stats, payloads = run_pipeline(tmp_path, "late", gallery_with_alice())
    types = [p["type"] for p in payloads]
    assert types == ["person_entered", "identity_resolved"]
    assert payloads[1]["identity"] == "alice"
    assert payloads[1]["data"]["entry_event_id"] == payloads[0]["event_id"]


def test_small_face_reports_face_too_small_not_identity(tmp_path):
    stats, payloads = run_pipeline(tmp_path, "small", gallery_with_alice())
    assert len(payloads) == 1
    assert payloads[0]["identity"] is None
    assert payloads[0]["data"]["unresolved_reason"] == "face_too_small"


def test_one_face_never_identifies_two_overlapping_tracks(tmp_path):
    # Unit-level at the enrichment boundary (ByteTrack itself cannot hold two
    # near-identical overlapping synthetic boxes apart — verified by probe —
    # so exclusivity is asserted where face->track assignment actually lives).
    from vision_platform.types import TrackObservation

    out = tmp_path / "events.jsonl"
    runner = PipelineRunner(spec(), detector=FakeDetector([]),
                            encoder=FakeEncoder([[face_inside(0.32)]], model_id="fake/1"),
                            gallery=gallery_with_alice(),
                            emitter=FileEmitter(Outbox(tmp_path / "o.db"), out),
                            machine_id="test-box")
    obs = [
        TrackObservation(camera_id="cam", ts=0.0, track_id=1,
                         bbox=(270, 740, 330, 900), frame_size=(1000, 1000), class_id=0),
        TrackObservation(camera_id="cam", ts=0.0, track_id=2,
                         bbox=(310, 740, 370, 900), frame_size=(1000, 1000), class_id=0),
    ]
    enriched, _ = runner._enrich(obs, FRAME)  # one face whose center is inside both
    assert sum(o.identity == "alice" for o in enriched) == 1
    assert sum(o.identity is None for o in enriched) == 1


def test_lower_quality_matching_face_still_resolves(tmp_path):
    # First seen face: excellent detector score, not Alice. A later, slightly
    # lower-score Alice face must still be tried (detector score is not a
    # proxy for matchability).
    from vision_platform.types import TrackObservation

    def hit(seed, quality):
        emb = np.zeros(128, dtype=np.float32)
        emb[seed] = 1.0
        return FaceHit(bbox=(295, 750, 345, 800), embedding=emb, quality=quality)

    out = tmp_path / "events.jsonl"
    runner = PipelineRunner(spec(), detector=FakeDetector([]),
                            encoder=FakeEncoder([[hit(1, 0.99)], [hit(0, 0.80)]],
                                                model_id="fake/1"),
                            gallery=gallery_with_alice(),
                            emitter=FileEmitter(Outbox(tmp_path / "o.db"), out),
                            machine_id="test-box")
    obs = [TrackObservation(camera_id="cam", ts=0.0, track_id=1,
                            bbox=(270, 740, 330, 900), frame_size=(1000, 1000), class_id=0)]
    e1, _ = runner._enrich(obs, FRAME)
    assert e1[0].identity is None
    e2, _ = runner._enrich(obs, FRAME)
    assert e2[0].identity == "alice"


def test_dwell_example_pipeline_runs_without_face_nodes(tmp_path):
    spec2 = load_pipeline(ROOT / "pipelines" / "dwell_alert.yaml")
    frames = [(float(t), FRAME) for t in range(0, 130, 10)]
    dets = [[person_at(0.5, 0.5)] for _ in frames]
    out = tmp_path / "events.jsonl"
    runner = PipelineRunner(spec2, detector=FakeDetector(dets), encoder=None,
                            gallery=None, emitter=FileEmitter(Outbox(tmp_path / "o.db"), out),
                            machine_id="test-box")
    runner.run(frames)
    payloads = [json.loads(line) for line in out.read_text().splitlines()]
    assert [p["type"] for p in payloads] == ["loitering"]
    assert payloads[0]["identity"] is None and "unresolved_reason" not in payloads[0]["data"]


def test_registry_conformance_every_manifest_constructible():
    # Every registered node type must be handled by the runner factory.
    from vision_platform.pipeline import RUNNER_NODE_TYPES
    assert set(REGISTRY) == set(RUNNER_NODE_TYPES)
