from pathlib import Path

import pytest
import yaml

from vision_platform.pipeline import REGISTRY, PipelineSpecError, load_pipeline

ROOT = Path(__file__).parent.parent


def _write(tmp_path, doc):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


BASE = {
    "name": "testpipe",
    "camera_id": "cam1",
    "source": {"analytics_fps": 5},
    "runtime": {"profile": "dev-cpu", "providers": ["CPUExecutionProvider"]},
    "zones": {"door": [[0, 0], [0.2, 0], [0.2, 1], [0, 1]],
              "room": [[0.2, 0], [1, 0], [1, 1], [0.2, 1]]},
    "nodes": [
        {"name": "src", "type": "source.file", "config": {}},
        {"name": "det", "type": "detect.persons", "config": {}},
        {"name": "trk", "type": "track.bytetrack", "config": {}},
        {"name": "seq", "type": "zone.sequence",
         "config": {"order": ["door", "room"], "within_s": 30, "event_type": "person_entered"}},
        {"name": "deb", "type": "debounce", "config": {"cooldown_s": 300}},
        {"name": "out", "type": "emit.file", "config": {}},
    ],
}


def test_shipped_example_pipelines_load():
    for name in ("attendance_poc.yaml", "dwell_alert.yaml"):
        spec = load_pipeline(ROOT / "pipelines" / name)
        assert spec.nodes[-1].type.startswith("emit.")


def test_minimal_pipeline_loads(tmp_path):
    spec = load_pipeline(_write(tmp_path, BASE))
    assert spec.name == "testpipe" and len(spec.nodes) == 6
    assert spec.zones["door"][0] == (0, 0)


def test_unknown_node_type_named(tmp_path):
    doc = {**BASE, "nodes": [{"name": "x", "type": "magic.box", "config": {}}]}
    with pytest.raises(PipelineSpecError, match="magic.box"):
        load_pipeline(_write(tmp_path, doc))


def test_missing_zone_reference_named(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"][3]["config"]["order"] = ["door", "nowhere"]
    with pytest.raises(PipelineSpecError, match="nowhere"):
        load_pipeline(_write(tmp_path, doc))


def test_duplicate_node_names_rejected(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"][1]["name"] = "src"
    with pytest.raises(PipelineSpecError, match="src"):
        load_pipeline(_write(tmp_path, doc))


def test_cycle_rejected(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"][0]["inputs"] = ["out"]
    with pytest.raises(PipelineSpecError, match="cycle|cyclic"):
        load_pipeline(_write(tmp_path, doc))


def test_port_mismatch_rejected(tmp_path):
    doc = json_clone(BASE)
    # sequence consumes Observations; wire it directly to the Frame source
    doc["nodes"][3]["inputs"] = ["src"]
    with pytest.raises(PipelineSpecError, match="port"):
        load_pipeline(_write(tmp_path, doc))


def test_unsupported_backend_profile_rejected(tmp_path):
    doc = json_clone(BASE)
    doc["runtime"]["profile"] = "jetson-deepstream"  # detect.persons has no deepstream backend yet
    with pytest.raises(PipelineSpecError, match="backend|profile"):
        load_pipeline(_write(tmp_path, doc))


def test_missing_emit_node_rejected(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"] = doc["nodes"][:-1]
    with pytest.raises(PipelineSpecError, match="emit"):
        load_pipeline(_write(tmp_path, doc))


def test_empty_sequence_config_rejected_at_validation(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"][3]["config"] = {}
    with pytest.raises(PipelineSpecError, match="order"):
        load_pipeline(_write(tmp_path, doc))


def test_first_node_cannot_consume_frame(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"] = doc["nodes"][1:]  # starts at detect.persons, no Frame producer
    with pytest.raises(PipelineSpecError, match="port"):
        load_pipeline(_write(tmp_path, doc))


def test_face_match_without_encode_rejected(tmp_path):
    doc = json_clone(BASE)
    doc["nodes"].insert(3, {"name": "match", "type": "face.match",
                            "config": {"match_threshold": 0.5}})
    with pytest.raises(PipelineSpecError, match="face.encode"):
        load_pipeline(_write(tmp_path, doc))


def test_pipeline_version_is_carried(tmp_path):
    doc = json_clone(BASE)
    doc["version"] = 7
    assert load_pipeline(_write(tmp_path, doc)).version == 7


def test_profile_provider_mismatch_rejected(tmp_path):
    doc = json_clone(BASE)
    doc["runtime"] = {"profile": "jetson-ort", "providers": ["CPUExecutionProvider"]}
    with pytest.raises(PipelineSpecError, match="provider"):
        load_pipeline(_write(tmp_path, doc))


def test_registry_manifests_have_known_port_kinds():
    kinds = {"None", "Frame", "Detections", "Observations", "Events"}
    for m in REGISTRY.values():
        assert m.consumes in kinds and m.produces in kinds and m.backends


def json_clone(doc):
    import copy
    return copy.deepcopy(doc)
