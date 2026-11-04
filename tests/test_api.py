from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from backend import config
from backend.comfy_runner import GenerationResult
from backend.main import app


def _png(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 70, 60)).save(buf, format="PNG")
    return buf.getvalue()


def test_status_shape(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("online", "ready", "faceid", "insightface", "busy", "queue_count"):
        assert key in data


def test_preflight_detects_face(client, preset_path):
    resp = client.post(
        "/api/preflight",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["face_detected"] is True
    assert "skin_tone" in data


def test_preflight_rejects_no_face(client):
    resp = client.post(
        "/api/preflight",
        files={"image": ("gray.png", _png((512, 512)), "image/png")},
    )
    assert resp.status_code == 422


def test_invalid_sliders_are_rejected(client, preset_path):
    resp = client.post(
        "/api/generate",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"prompt": "studio", "aspect": "4:5", "sliders": "not-json"},
    )
    assert resp.status_code == 422


def test_generate_success_with_mocked_runner(client, preset_path, monkeypatch):
    async def fake_status(force=False):
        return {"online": True, "ready": True, "error": "", "checkpoint": "Juggernaut"}

    async def fake_runner(*args, **kwargs):
        return GenerationResult(
            image_bytes=_png((896, 1152)),
            seed=123,
            checkpoint="Juggernaut",
            width=896,
            height=1152,
            elapsed=1.0,
            face_weights={"weight": 1.15, "v2": 0.65, "lora": 0.8},
            hires=False,
        )

    def fake_verify(ref, gen, lab):
        return {
            "ok": True,
            "identity_similarity": 0.92,
            "face_detected": True,
            "face_count": 1,
            "det_score": 0.9,
            "texture": 150.0,
            "skin_delta_e": 4.0,
            "skin_delta_chroma": 3.0,
            "reason": "",
        }

    monkeypatch.setattr("backend.main.check_comfy_status", fake_status)
    monkeypatch.setattr("backend.main.run_comfy_generation", fake_runner)
    monkeypatch.setattr("backend.main.verify", fake_verify)

    resp = client.post(
        "/api/generate",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"prompt": "studio portrait, elegant outfit", "aspect": "4:5", "sliders": "{}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["image_base64"].startswith("data:image/png;base64,")
    assert data["identity_verified"] is True
    assert data["seed"] == 123
    assert data["rescue_applied"] is False


def test_low_identity_is_rejected_without_rescue(client, preset_path, monkeypatch):
    calls = {"n": 0}

    async def fake_status(force=False):
        return {"online": True, "ready": True, "error": "", "checkpoint": "Juggernaut"}

    async def fake_runner(*args, **kwargs):
        calls["n"] += 1
        return GenerationResult(
            image_bytes=_png((896, 1152)),
            seed=42,
            checkpoint="Juggernaut",
            width=896,
            height=1152,
            elapsed=1.0,
            face_weights={},
            hires=False,
        )

    def fake_verify(ref, gen, lab):
        return {
            "ok": False,
            "identity_similarity": 0.40,
            "face_detected": True,
            "face_count": 1,
            "det_score": 0.9,
            "texture": 150.0,
            "reference_texture": 50.0,
            "skin_delta_e": 4.0,
            "skin_delta_chroma": 3.0,
            "reason": "Identity similarity 0.400 is below the gate.",
        }

    monkeypatch.setattr("backend.main.check_comfy_status", fake_status)
    monkeypatch.setattr("backend.main.run_comfy_generation", fake_runner)
    monkeypatch.setattr("backend.main.verify", fake_verify)
    monkeypatch.setattr(config, "ENABLE_IDENTITY_RESCUE", False)

    resp = client.post(
        "/api/generate",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"prompt": "studio portrait", "aspect": "4:5", "sliders": "{}"},
    )
    assert resp.status_code == 422
    assert calls["n"] == 2


def _rescue_metrics(similarity: float, chroma: float = 2.5) -> dict:
    return {
        "ok": True,
        "identity_similarity": similarity,
        "face_detected": True,
        "face_count": 1,
        "det_score": 0.92,
        "texture": 60.0,
        "reference_texture": 200.0,
        "skin_delta_e": 3.0,
        "skin_delta_chroma": chroma,
        "reason": "",
    }


def _patch_generation(monkeypatch, sim, chroma, calls=None):
    state = {"verify_calls": 0}

    async def fake_status(force=False):
        return {"online": True, "ready": True, "error": "", "checkpoint": "Cyber"}

    async def fake_runner(*args, **kwargs):
        if calls is not None:
            calls["n"] += 1
        return GenerationResult(
            image_bytes=_png((896, 1152)),
            seed=9,
            checkpoint="Cyber",
            width=896,
            height=1152,
            elapsed=1.0,
            face_weights={},
            hires=False,
        )

    def fake_verify(ref, gen, lab):
        # The first call is the raw FaceID pass (fails); later calls verify the
        # rescued/graded render, so they must reflect the rescue metrics.
        state["verify_calls"] += 1
        if state["verify_calls"] == 1:
            return {
                "ok": False,
                "identity_similarity": 0.30,
                "face_detected": True,
                "face_count": 1,
                "det_score": 0.9,
                "texture": 40.0,
                "reference_texture": 200.0,
                "skin_delta_e": 20.0,
                "skin_delta_chroma": 25.0,
                "reason": "low",
            }
        return _rescue_metrics(sim, chroma)

    def fake_rescue(gen_bytes, profile):
        return _png((896, 1152)), _rescue_metrics(sim, chroma)

    monkeypatch.setattr("backend.main.check_comfy_status", fake_status)
    monkeypatch.setattr("backend.main.run_comfy_generation", fake_runner)
    monkeypatch.setattr("backend.main.verify", fake_verify)
    monkeypatch.setattr("backend.main._run_rescue", fake_rescue)


def test_rescue_high_similarity_is_returned(client, preset_path, monkeypatch):
    _patch_generation(monkeypatch, sim=0.93, chroma=2.5)
    resp = client.post(
        "/api/generate",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"prompt": "studio portrait, elegant outfit", "aspect": "4:5", "sliders": "{}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["rescue_applied"] is True
    assert data["identity_match_percent"] == 93.0
    assert data["skin_delta_chroma"] == 2.5
    assert data["rescue_attempts"] == 1
    assert "face_enhanced" in data


def test_rescue_below_gate_is_rejected(client, preset_path, monkeypatch):
    calls = {"n": 0}
    _patch_generation(monkeypatch, sim=0.84, chroma=3.0, calls=calls)
    resp = client.post(
        "/api/generate",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"prompt": "studio portrait", "aspect": "4:5", "sliders": "{}"},
    )
    assert resp.status_code == 422
    assert "84%" in resp.json()["detail"]
    assert calls["n"] == 2


def test_queue_async_submit_and_poll(client, preset_path, monkeypatch):
    async def fake_status(force=False):
        return {"online": True, "ready": True, "error": "", "checkpoint": "Juggernaut"}

    async def fake_runner(*args, **kwargs):
        return GenerationResult(
            image_bytes=_png((896, 1152)),
            seed=456,
            checkpoint="Juggernaut",
            width=896,
            height=1152,
            elapsed=0.5,
            face_weights={"weight": 1.15, "v2": 0.65, "lora": 0.8},
            hires=False,
        )

    def fake_verify(ref, gen, lab):
        return {
            "ok": True,
            "identity_similarity": 0.94,
            "face_detected": True,
            "face_count": 1,
            "det_score": 0.95,
            "texture": 160.0,
            "skin_delta_e": 3.0,
            "skin_delta_chroma": 2.0,
            "reason": "",
        }

    monkeypatch.setattr("backend.main.check_comfy_status", fake_status)
    monkeypatch.setattr("backend.main.run_comfy_generation", fake_runner)
    monkeypatch.setattr("backend.main.verify", fake_verify)

    resp = client.post(
        "/api/generate?wait=false",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"prompt": "studio portrait", "aspect": "4:5", "sliders": "{}"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    job_id = data["job_id"]

    import time
    done = False
    for _ in range(50):
        poll = client.get(f"/api/jobs/{job_id}")
        assert poll.status_code == 200
        pdata = poll.json()
        if pdata["status"] == "completed":
            assert pdata["result"]["seed"] == 456
            done = True
            break
        time.sleep(0.1)
    assert done, "Async job did not complete in time"


def test_dress_swap_validation_errors(client, preset_path):
    # Empty prompt is rejected
    resp = client.post(
        "/api/dress-swap",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"dress_prompt": "   ", "category": "one-pieces"},
    )
    assert resp.status_code == 422


def test_dress_swap_success_mocked(client, preset_path, monkeypatch):
    import numpy as np

    async def fake_status(force=False):
        return {"online": True, "ready": True, "error": "", "checkpoint": "CyberRealisticXLPlay"}

    async def fake_inpaint(*args, **kwargs):
        return GenerationResult(
            image_bytes=_png((896, 1152)),
            seed=789,
            checkpoint="CyberRealisticXLPlay",
            width=896,
            height=1152,
            elapsed=0.8,
            face_weights={"weight": 1.15, "v2": 0.65, "lora": 0.8},
            hires=False,
        )

    def fake_verify(ref, gen, lab):
        return {
            "ok": True,
            "identity_similarity": 0.91,
            "face_detected": True,
            "face_count": 1,
            "det_score": 0.92,
            "texture": 180.0,
            "skin_delta_e": 2.5,
            "skin_delta_chroma": 1.8,
            "reason": "",
        }

    class FakeClothingParser:
        def extract_clothing_mask(self, raw, category="one-pieces"):
            # Return dummy 512x512 mask with > 500 white pixels
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[200:400, 100:400] = 255
            protect = np.zeros((512, 512), dtype=np.uint8)
            return mask, protect

    monkeypatch.setattr("backend.main.check_comfy_status", fake_status)
    monkeypatch.setattr("backend.main.run_clothing_inpaint", fake_inpaint)
    monkeypatch.setattr("backend.main.verify", fake_verify)
    monkeypatch.setattr("backend.clothing_parser.ClothingParser.get_instance", lambda: FakeClothingParser())

    resp = client.post(
        "/api/dress-swap",
        files={"image": ("ref.jpg", preset_path.read_bytes(), "image/jpeg")},
        data={"dress_prompt": "royal blue velvet evening dress", "category": "one-pieces"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["mode"] == "dress-swap"
    assert data["category"] == "one-pieces"
    assert data["seed"] == 789
    assert data["identity_verified"] is True
    assert data["image_base64"].startswith("data:image/png;base64,")

