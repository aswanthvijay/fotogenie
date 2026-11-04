from __future__ import annotations

import pytest

from backend import config, comfy_runner


def test_cyberrealistic_is_the_default_primary(monkeypatch):
    monkeypatch.setattr(config, "PREFERRED_CHECKPOINT", "")
    monkeypatch.setattr(config, "ALLOW_NSFW_CHECKPOINT", True)
    chosen = comfy_runner.choose_checkpoint(
        [
            "RealVisXL_V5.0_fp16.safetensors",
            "CyberRealisticXLPlay_V10.0_FP16.safetensors",
            "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        ]
    )
    assert chosen == "CyberRealisticXLPlay_V10.0_FP16.safetensors"


def test_checkpoint_preference_is_not_list_order(monkeypatch):
    monkeypatch.setattr(config, "PREFERRED_CHECKPOINT", "")
    monkeypatch.setattr(config, "ALLOW_NSFW_CHECKPOINT", False)
    chosen = comfy_runner.choose_checkpoint(
        [
            "CyberRealisticXLPlay_V10.0_FP16.safetensors",
            "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            "RealVisXL_V5.0_fp16.safetensors",
        ]
    )
    assert chosen == "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"


def test_nsfw_checkpoint_requires_opt_in(monkeypatch):
    monkeypatch.setattr(config, "PREFERRED_CHECKPOINT", "")
    monkeypatch.setattr(config, "ALLOW_NSFW_CHECKPOINT", False)
    with pytest.raises(comfy_runner.ComfyGraphError):
        comfy_runner.choose_checkpoint(["CyberRealisticXLPlay_V10.0_FP16.safetensors"])


def test_build_workflow_contains_faceid_and_hires():
    workflow, save_node, out_w, out_h = comfy_runner.build_workflow(
        checkpoint="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        image_name="ref.png",
        positive_prompt="portrait",
        negative_prompt="blurry",
        width=896,
        height=1152,
        seed=42,
        weights={"weight": 1.15, "v2": 0.65, "lora": 0.8},
        hires=True,
        hires_factor=1.25,
        hires_steps=18,
        hires_denoise=0.35,
    )
    assert workflow["4"]["class_type"] == "IPAdapterUnifiedLoaderFaceID"
    assert workflow["6"]["class_type"] == "IPAdapterFaceID"
    assert workflow["6"]["inputs"]["image"] == ["2", 0]
    assert workflow["6"]["inputs"]["weight"] == 1.15
    assert workflow["12"]["class_type"] == "LatentUpscaleBy"
    assert workflow["13"]["class_type"] == "KSampler"
    assert save_node == "15"
    assert (out_w, out_h) == (1120, 1440)


def test_build_workflow_without_hires():
    workflow, save_node, out_w, out_h = comfy_runner.build_workflow(
        checkpoint="RealVisXL_V5.0_fp16.safetensors",
        image_name="ref.png",
        positive_prompt="portrait",
        negative_prompt="blurry",
        width=768,
        height=1344,
        seed=1,
        weights={},
        hires=False,
        hires_factor=1.0,
        hires_steps=18,
        hires_denoise=0.35,
    )
    assert save_node == "11"
    assert "12" not in workflow
    assert (out_w, out_h) == (768, 1344)


def test_detail_workflow_is_faceid_conditioned():
    workflow, save_node = comfy_runner.build_detail_workflow(
        reference_name="reference.png",
        crop_name="crop.png",
        positive_prompt="portrait",
        negative_prompt="blurry",
        weights={"weight": 1.25, "v2": 0.75, "lora": 0.95},
        denoise=0.45,
        steps=22,
        seed=7,
        checkpoint="CyberRealisticXLPlay_V10.0_FP16.safetensors",
    )
    assert workflow["2"]["inputs"]["image"] == "reference.png"
    assert workflow["3"]["inputs"]["image"] == "crop.png"
    assert workflow["10"]["class_type"] == "VAEEncode"
    assert workflow["10"]["inputs"]["pixels"] == ["3", 0]
    assert workflow["9"]["inputs"]["denoise"] == 0.45
    assert workflow["9"]["inputs"]["latent_image"] == ["10", 0]
    assert workflow["6"]["class_type"] == "IPAdapterFaceID"
    assert workflow["6"]["inputs"]["image"] == ["2", 0]
    assert workflow["6"]["inputs"]["weight"] == 1.25
    assert workflow["12"]["class_type"] == "SaveImage"
    assert save_node == "12"


def test_upscale_workflow_targets_an_explicit_size():
    workflow, save_node = comfy_runner.build_upscale_workflow(
        "img.png", 1200, 2100, "4x-UltraSharp.pth"
    )
    assert workflow["2"]["class_type"] == "UpscaleModelLoader"
    assert workflow["2"]["inputs"]["model_name"] == "4x-UltraSharp.pth"
    assert workflow["3"]["class_type"] == "ImageUpscaleWithModel"
    assert workflow["4"]["class_type"] == "ImageScale"
    assert workflow["4"]["inputs"]["width"] == 1200
    assert workflow["4"]["inputs"]["height"] == 2100
    assert save_node == "5"


def test_hires_plan_respects_pixel_budget():
    hires, factor, out_w, out_h = comfy_runner._hires_plan(1344, 768)
    assert hires is True
    assert out_w * out_h <= config.MAX_HIRES_PIXELS + 4096
    assert factor >= 1.05


def test_error_extraction_is_short():
    status = {
        "status_str": "error",
        "messages": [["execution_error", {"node_type": "KSampler", "exception_message": "boom"}]],
    }
    message = comfy_runner._error_from_status(status)
    assert "KSampler" in message
    assert len(message) <= 400
