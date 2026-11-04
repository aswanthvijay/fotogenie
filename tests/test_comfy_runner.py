"""Regression tests for graceful ComfyUI failure handling.

These lock in the contract that a post-processing failure (the optional detail
pass and the final upscale) never discards an accepted render, and that
/api/status reports every readiness blocker rather than only the last one
checked.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
from PIL import Image

from backend import comfy_runner, config


def _png(size=(512, 512)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 96, 84)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeFace:
    def __init__(self, bbox):
        self.bbox = bbox


class _FakeEngine:
    def largest(self, bgr):
        h, w = bgr.shape[:2]
        return _FakeFace([int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)])


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


class _UploadClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        return _Response(200, {"name": "uploaded.png"})


class _StatusClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, *args, **kwargs):
        if "CheckpointLoaderSimple" in url:
            return _Response(
                200,
                {
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {
                                "ckpt_name": [["RealVisXL_V5.0_fp16.safetensors"]]
                            }
                        }
                    }
                },
            )
        if "IPAdapter" in url:
            return _Response(404, {})
        if "system_stats" in url:
            return _Response(
                200,
                {
                    "devices": [
                        {
                            "name": "RTX 3060",
                            "vram_total": 6442450944,
                            "vram_free": 5000000000,
                        }
                    ]
                },
            )
        return _Response(404, {})


async def _fake_status(force=False):
    return {
        "online": True,
        "ready": True,
        "checkpoint": "RealVisXL_V5.0_fp16.safetensors",
        "vram_free_gb": 10.0,
    }


def _patch_uploads(monkeypatch):
    monkeypatch.setattr(comfy_runner.httpx, "AsyncClient", _UploadClient)


async def test_face_detail_failure_returns_original(monkeypatch):
    async def boom(client, workflow, save_node, timeout):
        raise comfy_runner.ComfyGraphError("boom")

    monkeypatch.setattr(comfy_runner, "check_comfy_status", _fake_status)
    monkeypatch.setattr(
        "backend.face_engine.FaceEngine.get", staticmethod(lambda: _FakeEngine())
    )
    _patch_uploads(monkeypatch)
    monkeypatch.setattr(comfy_runner, "_run_graph", boom)

    src = _png()
    out, meta = await comfy_runner.run_face_detail(src, src, "pos", "neg")

    assert out is src
    assert meta["applied"] is False
    assert "boom" in meta["reason"]


async def test_final_upscale_http_error_returns_original(monkeypatch):
    async def boom(client, workflow, save_node, timeout):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(config, "ENABLE_FINAL_UPSCALE", True)
    monkeypatch.setattr(comfy_runner, "check_comfy_status", _fake_status)
    monkeypatch.setattr(comfy_runner, "_upscale_model_path", lambda: Path(__file__))
    _patch_uploads(monkeypatch)
    monkeypatch.setattr(comfy_runner, "_run_graph", boom)

    src = _png()
    out, meta = await comfy_runner.run_final_upscale(src)

    assert out is src
    assert meta["applied"] is False
    assert "timed out" in meta["reason"]


async def test_status_reports_every_blocker(monkeypatch):
    monkeypatch.setattr(comfy_runner.httpx, "AsyncClient", _StatusClient)
    monkeypatch.setattr(config, "PREFERRED_CHECKPOINT", "")
    monkeypatch.setattr(comfy_runner, "BUFFALO_FILES", ())
    monkeypatch.setattr(
        comfy_runner,
        "_missing_models",
        lambda: ["models/insightface/models/buffalo_l/det_10g.onnx"],
    )
    monkeypatch.setattr(comfy_runner, "_STATUS_CACHE", {"at": 0.0, "value": None})

    status = await comfy_runner.check_comfy_status(force=True)

    assert status["online"] is True
    assert status["faceid"] is False
    assert status["ready"] is False
    assert "Missing model files" in status["error"]
    assert "IP-Adapter FaceID nodes" in status["error"]


def test_build_clothing_inpaint_workflow():
    wf, save_node = comfy_runner.build_clothing_inpaint_workflow(
        checkpoint="CyberRealisticXLPlay_V10.0_FP16.safetensors",
        person_image_name="person.png",
        mask_image_name="mask.png",
        face_crop_name="face.png",
        positive_prompt="silk dress",
        negative_prompt="blurry",
        seed=42,
        weights={"weight": 1.15, "v2": 0.65, "lora": 0.8},
        denoise=0.95,
        steps=30,
        grow_mask_by=8,
    )
    assert save_node == "12"
    assert "9" in wf  # VAEEncodeForInpaint
    assert wf["9"]["class_type"] == "VAEEncodeForInpaint"
    assert wf["9"]["inputs"]["grow_mask_by"] == 8
    assert wf["3"]["class_type"] == "LoadImageMask"
    assert wf["3"]["inputs"]["channel"] == "red"
    assert wf["6"]["class_type"] == "IPAdapterFaceID"
    assert wf["10"]["class_type"] == "KSampler"
    assert wf["10"]["inputs"]["denoise"] == 0.95

