from vision_platform.bench import LatencyRecorder, estimate_cameras_per_box


def test_recorder_mean_and_p95():
    r = LatencyRecorder()
    for ms in range(1, 101):
        r.record("detect.persons", float(ms))
    s = r.summary()["detect.persons"]
    assert s["count"] == 100
    assert abs(s["mean_ms"] - 50.5) < 1e-9
    assert s["p95_ms"] == 95.0


def test_estimate_detect_bound():
    summary = {
        "detect.persons": {"count": 10, "mean_ms": 100.0, "p95_ms": 110.0},
        "track.bytetrack": {"count": 10, "mean_ms": 0.0, "p95_ms": 0.0},
        "face.encode+match": {"count": 10, "mean_ms": 0.0, "p95_ms": 0.0},
    }
    est = estimate_cameras_per_box(summary, analytics_fps=5, face_cameras_fraction=1.0)
    assert est["cameras"] == 2
    assert est["bottleneck_node"] == "detect.persons"
    assert any("speedup=1.0" in a for a in est["assumptions"])


def test_estimate_mixed_fleet_weights_face_cost():
    summary = {
        "detect.persons": {"count": 10, "mean_ms": 10.0, "p95_ms": 12.0},
        "face.encode+match": {"count": 10, "mean_ms": 30.0, "p95_ms": 40.0},
    }
    all_face = estimate_cameras_per_box(summary, analytics_fps=5, face_cameras_fraction=1.0)
    half_face = estimate_cameras_per_box(summary, analytics_fps=5, face_cameras_fraction=0.5)
    assert all_face["cameras"] == 5    # 1000 / (5 * 40)
    assert half_face["cameras"] == 8   # 1000 / (5 * (10 + 0.5*30)) = 8


def test_estimate_speedup_scales_linearly_and_is_named():
    summary = {"detect.persons": {"count": 1, "mean_ms": 100.0, "p95_ms": 100.0}}
    est = estimate_cameras_per_box(summary, analytics_fps=5, face_cameras_fraction=0.0,
                                   accelerator_speedup=4.0)
    assert est["cameras"] == 8
    assert any("speedup=4.0" in a for a in est["assumptions"])
