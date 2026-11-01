import json
from pathlib import Path

from vision_platform.events import Debouncer, DwellDetector, Gate, SequenceDetector, ZoneSet
from vision_platform.types import TrackObservation

ZONES = {
    "door": [(0, 0), (0.2, 0), (0.2, 1), (0, 1)],
    "hall": [(0.2, 0), (0.4, 0), (0.4, 1), (0.2, 1)],
    "room": [(0.4, 0), (1, 0), (1, 1), (0.4, 1)],
}
ZS = ZoneSet(ZONES)


def obs(track=1, x=0.1, ts=0.0, identity=None):
    # bbox whose foot point normalizes to (x, 0.9) in a 1000x1000 frame
    px = x * 1000
    return TrackObservation(camera_id="cam", ts=ts, track_id=track,
                            bbox=(px - 10, 860, px + 10, 900), frame_size=(1000, 1000),
                            class_id=0, identity=identity)


def feed(det, points):
    events = []
    for o in points:
        ev = det.update(o, ZS.zones_of(o))
        if ev:
            events.append(ev)
    return events


def test_sequence_door_hall_room_emits_once():
    det = SequenceDetector(order=["door", "hall", "room"], within_s=30, event_type="person_entered")
    evs = feed(det, [obs(x=0.1, ts=0), obs(x=0.3, ts=5), obs(x=0.6, ts=10), obs(x=0.7, ts=12)])
    assert len(evs) == 1
    assert evs[0].type == "person_entered"
    assert evs[0].data == {"from_zone": "door", "to_zone": "room", "via": ["hall"]}


def test_sequence_timeout_emits_nothing():
    det = SequenceDetector(order=["door", "hall", "room"], within_s=30, event_type="person_entered")
    evs = feed(det, [obs(x=0.1, ts=0), obs(x=0.6, ts=31)])
    assert evs == []


def test_sequence_reverse_and_midroom_birth_emit_nothing():
    det = SequenceDetector(order=["door", "hall", "room"], within_s=30, event_type="person_entered")
    evs = feed(det, [obs(x=0.6, ts=0), obs(x=0.3, ts=5), obs(x=0.1, ts=10), obs(x=0.3, ts=15), obs(x=0.6, ts=20)])
    assert evs == []  # born in room: permanently ineligible


def test_debouncer_suppresses_flicker_repeats():
    deb = Debouncer(cooldown_s=300)
    det = SequenceDetector(order=["door", "room"], within_s=30, event_type="person_entered")
    kept = []
    for cycle in range(3):
        base = cycle * 20
        for o in [obs(x=0.1, ts=base), obs(x=0.6, ts=base + 5)]:
            ev = det.update(o, ZS.zones_of(o))
            if ev and deb.allow(ev):
                kept.append(ev)
    assert len(kept) == 1


def test_debouncer_keys_on_identity_across_tracks():
    deb = Debouncer(cooldown_s=300)
    det = SequenceDetector(order=["door", "room"], within_s=30, event_type="person_entered")
    kept = []
    for track, base in ((1, 0), (2, 100)):
        for o in [obs(track=track, x=0.1, ts=base, identity="alice"),
                  obs(track=track, x=0.6, ts=base + 5, identity="alice")]:
            ev = det.update(o, ZS.zones_of(o))
            if ev and deb.allow(ev):
                kept.append(ev)
    assert len(kept) == 1  # second track, same identity, still in cooldown


def test_dwell_emits_once_after_threshold():
    det = DwellDetector(zone="room", min_dwell_s=60, event_type="loitering")
    evs = []
    for ts in (0, 30, 59, 60, 90, 200):
        o = obs(x=0.6, ts=float(ts))
        ev = det.update(o, ZS.zones_of(o))
        if ev:
            evs.append(ev)
    assert len(evs) == 1 and evs[0].type == "loitering" and evs[0].data["zone"] == "room"


def test_gate_conditions_all_semantics():
    gate = Gate([{"field": "identity", "op": "is_null"},
                 {"field": "class_id", "op": "eq", "value": 0}])
    assert gate.passes(obs(identity=None)) is True
    assert gate.passes(obs(identity="alice")) is False


def test_sequence_advances_at_most_one_step_per_observation():
    # Overlapping zones: a track first seen inside both door and room must not
    # complete the sequence on its very first observation.
    both = frozenset({"door", "room"})
    det = SequenceDetector(order=["door", "room"], within_s=30, event_type="person_entered")
    assert det.update(obs(ts=0), both) is None
    assert det.update(obs(ts=1), both) is not None


def test_dwell_state_pruned_for_long_gone_tracks():
    det = DwellDetector(zone="room", min_dwell_s=60, event_type="loitering")
    o1 = obs(track=1, x=0.6, ts=0.0)
    det.update(o1, ZS.zones_of(o1))
    assert len(det._state) == 1
    o2 = obs(track=2, x=0.6, ts=10000.0)
    det.update(o2, ZS.zones_of(o2))
    assert 1 not in det._state  # pruned, unbounded growth avoided


def test_golden_fixture_entry_sequence():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "entry_sequence.json").read_text())
    zs = ZoneSet({k: [tuple(p) for p in v] for k, v in fixture["zones"].items()})
    det = SequenceDetector(order=fixture["order"], within_s=fixture["within_s"],
                           event_type=fixture["event_type"])
    got = []
    for raw in fixture["observations"]:
        o = TrackObservation(**{**raw, "bbox": tuple(raw["bbox"]), "frame_size": tuple(raw["frame_size"])})
        ev = det.update(o, zs.zones_of(o))
        if ev:
            got.append({"type": ev.type, "track_id": ev.track_id, "identity": ev.identity, "data": ev.data})
    assert got == fixture["expected_events"]
