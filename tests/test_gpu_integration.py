from __future__ import annotations

import base64

import pytest

from backend import config
from backend.face_guard import quality_ok, verify


@pytest.mark.gpu
def test_real_generation_passes_identity_and_face_quality(client, preset_path):
    resp = client.post(
        "/api/generate",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={
            "prompt": "professional studio portrait, wearing an elegant outfit, soft natural lighting",
            "aspect": "4:5",
            "sliders": "{}",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    raw = base64.b64decode(data["image_base64"].split(",", 1)[1])
    from backend.analyzer import analyze_reference

    profile = analyze_reference(preset_path.read_bytes())
    metrics = verify(profile.face_crop_png, raw, profile.skin_lab)
    assert metrics["ok"] is True
    assert metrics["identity_similarity"] >= config.RESCUE_MIN_SIM
    assert metrics["face_count"] >= 1
    assert quality_ok(
        metrics, metrics.get("reference_texture"), min_sim=config.RESCUE_MIN_SIM
    )
