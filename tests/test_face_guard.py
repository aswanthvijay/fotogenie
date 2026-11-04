from __future__ import annotations

import io

from PIL import Image

from backend.face_guard import face_metrics, identity_ok, quality_ok, verify


def _gray(size=(256, 256)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_self_similarity_is_high(face_profile):
    result = verify(face_profile.face_crop_png, face_profile.face_crop_png, face_profile.skin_lab)
    assert result["face_detected"] is True
    assert result["identity_similarity"] > 0.95
    assert result["ok"] is True
    assert result["face_count"] >= 1


def test_no_face_output_is_rejected(face_profile):
    result = verify(face_profile.face_crop_png, _gray(), face_profile.skin_lab)
    assert result["ok"] is False
    assert result["face_detected"] is False


def test_metrics_are_available(face_profile):
    metrics = face_metrics(face_profile.face_crop_png)
    assert metrics["face_detected"] is True
    assert metrics["texture"] > 0
    assert metrics["skin_lab"] is not None
    good = {
        "ok": True,
        "identity_similarity": 0.95,
        "face_count": 1,
        "det_score": 0.9,
        "texture": 150.0,
        "skin_delta_chroma": 3.0,
    }
    assert quality_ok(good)
    assert quality_ok({**good, "texture": 16.0})
    assert not quality_ok({**good, "texture": 10.0})
    assert not quality_ok({**good, "identity_similarity": 0.80})
    assert not quality_ok({**good, "skin_delta_chroma": 12.0})
    assert identity_ok(good)
    assert not identity_ok({**good, "identity_similarity": 0.80})
