from vision_platform.geometry import foot_point, point_in_polygon
from vision_platform.types import Detection, MachineEvent, TrackObservation

SQUARE = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]


def test_foot_point_is_bottom_center():
    assert foot_point((10.0, 20.0, 30.0, 60.0)) == (20.0, 60.0)


def test_point_inside_square():
    assert point_in_polygon((0.25, 0.25), SQUARE) is True


def test_point_outside_square():
    assert point_in_polygon((0.75, 0.25), SQUARE) is False


def test_point_on_boundary_counts_inside():
    assert point_in_polygon((0.5, 0.25), SQUARE) is True


def test_concave_polygon():
    # L-shaped (concave) doorway-like zone
    poly = [(0, 0), (0.6, 0), (0.6, 0.3), (0.3, 0.3), (0.3, 0.6), (0, 0.6)]
    assert point_in_polygon((0.15, 0.45), poly) is True
    assert point_in_polygon((0.45, 0.45), poly) is False  # in the notch


def test_types_are_constructible_with_defaults():
    det = Detection(bbox=(0, 0, 10, 20), score=0.9, class_id=0)
    obs = TrackObservation(camera_id="cam1", ts=1.0, track_id=7, bbox=det.bbox,
                           frame_size=(1920, 1080), class_id=0)
    assert obs.identity is None and obs.identity_score is None
    ev = MachineEvent(event_id="e1", type="person_entered", ts=2.0, camera_id="cam1",
                      track_id=7, identity=None, identity_score=None, data={})
    assert ev.type == "person_entered"
