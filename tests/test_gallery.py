import numpy as np
import pytest

from vision_platform.gallery import Gallery


def vec(*xs):
    v = np.zeros(128, dtype=np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def make_gallery():
    g = Gallery(model_id="test-model/1")
    g.enroll("alice", vec(1.0, 0.01))
    g.enroll("alice", vec(1.0, -0.01))
    g.enroll("bob", vec(0.0, 1.0))
    return g


def test_identify_returns_nearest_identity_and_score():
    hit = make_gallery().identify(vec(1.0, 0.02), threshold=0.5)
    assert hit is not None
    name, score = hit
    assert name == "alice" and score > 0.9


def test_identify_unknown_below_threshold():
    assert make_gallery().identify(vec(0.0, 0.0, 1.0), threshold=0.5) is None


def test_template_is_normalized_mean():
    g = make_gallery()
    cos = float(np.dot(g.templates["alice"], vec(1.0, 0.0)))
    assert cos > 0.999


def test_save_load_roundtrip(tmp_path):
    g = make_gallery()
    p = tmp_path / "g.npz"
    g.save(p)
    g2 = Gallery.load(p)
    assert g2.model_id == "test-model/1"
    assert g2.identify(vec(0.0, 1.0), 0.5)[0] == "bob"


def test_model_id_mismatch_raises_on_identify():
    g = make_gallery()
    with pytest.raises(ValueError, match="model_id"):
        g.identify_with_model(vec(1.0), model_id="other/9", threshold=0.5)


def test_save_load_exact_path_without_suffix(tmp_path):
    g = make_gallery()
    p = tmp_path / "gallery"  # no .npz suffix: save must write exactly here
    g.save(p)
    assert p.exists()
    assert Gallery.load(p).identify(vec(1.0), 0.5)[0] == "alice"


def test_corrupt_gallery_raises_value_error(tmp_path):
    p = tmp_path / "g.npz"
    p.write_bytes(b"not a zip at all")
    with pytest.raises(ValueError, match="gallery"):
        Gallery.load(p)


def test_enroll_rejects_zero_vector():
    with pytest.raises(ValueError):
        Gallery(model_id="m").enroll("x", np.zeros(128, dtype=np.float32))
