import json

from vision_platform.emit import FileEmitter, Outbox, WebhookEmitter, to_payload
from vision_platform.types import MachineEvent


def ev(event_id="e-1", identity="alice"):
    return MachineEvent(event_id=event_id, type="person_entered", ts=12.5,
                        camera_id="labkom_2", track_id=7, identity=identity,
                        identity_score=0.81,
                        data={"from_zone": "door", "to_zone": "room", "via": []})


def test_payload_envelope_exact_fields():
    p = to_payload(ev(), machine_id="jetson-1", pipeline="attendance_poc@1")
    assert p == {
        "schema": "vision.event/1", "event_id": "e-1", "type": "person_entered",
        "ts": 12.5, "machine_id": "jetson-1", "camera_id": "labkom_2",
        "pipeline": "attendance_poc@1", "track_id": 7, "identity": "alice",
        "identity_score": 0.81,
        "data": {"from_zone": "door", "to_zone": "room", "via": []}}


def test_outbox_put_idempotent_and_pending(tmp_path):
    box = Outbox(tmp_path / "o.db")
    box.put(ev(), "m", "p@1")
    box.put(ev(), "m", "p@1")
    pending = box.pending(10)
    assert len(pending) == 1 and pending[0]["event_id"] == "e-1"


def test_outbox_survives_restart(tmp_path):
    path = tmp_path / "o.db"
    Outbox(path).put(ev(), "m", "p@1")
    assert len(Outbox(path).pending(10)) == 1


def test_webhook_flush_failure_keeps_pending_then_sends_same_id(tmp_path):
    box = Outbox(tmp_path / "o.db")
    sent = []

    def fail_post(url, payload, timeout):
        return 500

    def ok_post(url, payload, timeout):
        sent.append(payload)
        return 200

    em = WebhookEmitter(box, "http://downstream/hook", post=fail_post)
    em.emit(ev(), "m", "p@1")
    assert len(box.pending(10)) == 1
    em2 = WebhookEmitter(box, "http://downstream/hook", post=ok_post)
    assert em2.flush() == 1
    assert box.pending(10) == []
    assert sent[0]["event_id"] == "e-1"


def test_file_emitter_writes_identical_payload(tmp_path):
    box = Outbox(tmp_path / "o.db")
    out = tmp_path / "events.jsonl"
    FileEmitter(box, out).emit(ev(identity=None), "m", "p@1")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["identity"] is None and payload["schema"] == "vision.event/1"
    assert box.pending(10) == []  # delivered locally
