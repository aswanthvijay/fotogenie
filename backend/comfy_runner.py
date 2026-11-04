"""ComfyUI HTTP client and workflow builder.

The graph is built explicitly so it can be unit-tested. FaceID is required:
a missing custom node or model raises, it never silently falls back to a
text-only generation that returns a generic face.
"""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import httpx
import numpy as np
from PIL import Image

from backend import config
from backend.logging import get_logger

logger = get_logger("fotogenie.comfy_runner")

MIN_IPADAPTER_PATTERN = "*faceid*plusv2*sdxl*.bin"
MIN_LORA_PATTERN = "*faceid*plusv2*sdxl*lora*.safetensors"
CLIP_VISION_PATTERN = "*clip-vit-h*"
# FaceAnalysis("buffalo_l") loads all five files; check all of them so
# /api/status reports a missing landmark/genderage model instead of only
# failing later when the engine is constructed.
BUFFALO_FILES = (
    "det_10g.onnx",
    "w600k_r50.onnx",
    "2d106det.onnx",
    "1k3d68.onnx",
    "genderage.onnx",
)


class ComfyError(RuntimeError):
    pass


class ComfyUnavailable(ComfyError):
    pass


class ComfyTimeout(ComfyError):
    pass


class ComfyGraphError(ComfyError):
    pass


class FaceIdUnavailable(ComfyError):
    pass


class OutOfMemory(ComfyGraphError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    image_bytes: bytes
    seed: int
    checkpoint: str
    width: int
    height: int
    elapsed: float
    face_weights: Dict[str, float]
    hires: bool


_STATUS_CACHE: dict = {"at": 0.0, "value": None}


def _find(directory: Path, pattern: str):
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def _missing_models() -> list:
    models = config.COMFY_ROOT / "models"
    missing = []
    if not _find(models / "ipadapter", MIN_IPADAPTER_PATTERN):
        missing.append("models/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin")
    if not _find(models / "loras", MIN_LORA_PATTERN):
        missing.append("models/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors")
    if not _find(models / "clip_vision", CLIP_VISION_PATTERN):
        missing.append("models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
    buffalo = config.INSIGHTFACE_ROOT / "models" / "buffalo_l"
    for name in BUFFALO_FILES:
        if not (buffalo / name).is_file():
            missing.append(f"models/insightface/models/buffalo_l/{name}")
    return missing


def choose_checkpoint(checkpoints: list) -> str:
    """Deterministic checkpoint selection; never free-form list order."""
    allowed = [c for c in checkpoints if not _is_nsfw_prone(c) or config.ALLOW_NSFW_CHECKPOINT]
    if config.PREFERRED_CHECKPOINT:
        if config.PREFERRED_CHECKPOINT not in checkpoints:
            raise ComfyGraphError(
                f"Configured checkpoint {config.PREFERRED_CHECKPOINT!r} is not installed."
            )
        if _is_nsfw_prone(config.PREFERRED_CHECKPOINT) and not config.ALLOW_NSFW_CHECKPOINT:
            raise ComfyGraphError(
                "The configured checkpoint is NSFW-prone; set FOTOGENIE_ALLOW_NSFW_CHECKPOINT=true to use it."
            )
        return config.PREFERRED_CHECKPOINT

    for preferred in config.CHECKPOINT_PREFERENCE:
        if preferred in checkpoints and preferred in allowed:
            return preferred
    if allowed:
        return allowed[0]
    raise ComfyGraphError(
        "No suitable SDXL checkpoint is installed. Expected CyberRealisticXLPlay, "
        "Juggernaut-XL v9 or RealVisXL V5; CyberRealisticXLPlay requires "
        "FOTOGENIE_ALLOW_NSFW_CHECKPOINT=true."
    )


def _is_nsfw_prone(name: str) -> bool:
    return "cyberrealistic" in name.lower() and "play" in name.lower()


async def check_comfy_status(force: bool = False) -> dict:
    """Query object_info and the filesystem; cached for the UI poller."""
    now = time.monotonic()
    cached = _STATUS_CACHE.get("value")
    if not force and cached is not None and now - _STATUS_CACHE["at"] < config.STATUS_CACHE_SECONDS:
        return dict(cached)

    result = {
        "online": False,
        "ready": False,
        "checkpoint": "",
        "checkpoints": [],
        "faceid": False,
        "insightface": False,
        "gpu_name": "",
        "vram_free_gb": None,
        "error": "",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
        try:
            ckpt_resp = await client.get(f"{config.COMFY_URL}/object_info/CheckpointLoaderSimple")
            if ckpt_resp.status_code != 200:
                result["error"] = f"ComfyUI returned HTTP {ckpt_resp.status_code} for the checkpoint loader."
                _STATUS_CACHE.update(at=now, value=dict(result))
                return dict(result)
            result["online"] = True
            data = ckpt_resp.json().get("CheckpointLoaderSimple", {})
            checkpoints = (
                data.get("input", {})
                .get("required", {})
                .get("ckpt_name", [[]])[0]
            )
            result["checkpoints"] = list(checkpoints or [])
            try:
                result["checkpoint"] = choose_checkpoint(result["checkpoints"])
            except ComfyError as exc:
                result["error"] = str(exc)
        except Exception as exc:
            result["error"] = f"ComfyUI is not reachable at {config.COMFY_URL}."
            _STATUS_CACHE.update(at=now, value=dict(result))
            return dict(result)

        try:
            loader = await client.get(f"{config.COMFY_URL}/object_info/IPAdapterUnifiedLoaderFaceID")
            faceid = await client.get(f"{config.COMFY_URL}/object_info/IPAdapterFaceID")
            result["faceid"] = loader.status_code == 200 and faceid.status_code == 200
        except Exception:
            result["faceid"] = False

        try:
            stats = await client.get(f"{config.COMFY_URL}/system_stats")
            if stats.status_code == 200:
                device = (stats.json().get("devices") or [{}])[0]
                result["gpu_name"] = device.get("name", "")
                total = float(device.get("vram_total") or 0)
                free = float(device.get("vram_free") or 0)
                result["vram_free_gb"] = round(free / (1024 ** 3), 2) if total else None
        except Exception:
            pass

    buffalo = config.INSIGHTFACE_ROOT / "models" / "buffalo_l"
    result["insightface"] = all((buffalo / name).is_file() for name in BUFFALO_FILES)
    missing = _missing_models()
    problems = [result["error"]] if result["error"] else []
    if missing:
        problems.append("Missing model files: " + ", ".join(missing[:3]))
    if not result["faceid"]:
        problems.append("IP-Adapter FaceID nodes are not installed in ComfyUI.")
    if problems:
        # Report every blocker, not just the last one checked. Overwriting the
        # earlier message hid the missing-model detail behind the node error.
        result["error"] = " ".join(problems)
    result["ready"] = bool(
        result["online"]
        and result["faceid"]
        and result["insightface"]
        and result["checkpoint"]
        and not missing
    )
    _STATUS_CACHE.update(at=now, value=dict(result))
    return dict(result)


def _aspect_size(aspect: str) -> Tuple[int, int]:
    size = config.ASPECT_MAP.get(aspect)
    if not size:
        raise ValueError(f"Unsupported aspect {aspect!r}; use one of {sorted(config.ASPECT_MAP)}.")
    return int(size[0]), int(size[1])


def _hires_plan(width: int, height: int) -> Tuple[bool, float, int, int]:
    if not config.ENABLE_HIRES:
        return False, 1.0, width, height
    factor = min(config.HIRES_FACTOR, (config.MAX_HIRES_PIXELS / float(width * height)) ** 0.5)
    if factor < 1.05:
        return False, 1.0, width, height
    out_w = int(round(width * factor))
    out_h = int(round(height * factor))
    return True, round(factor, 4), out_w, out_h


def build_workflow(
    checkpoint: str,
    image_name: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    weights: Dict[str, float],
    hires: bool,
    hires_factor: float,
    hires_steps: int,
    hires_denoise: float,
) -> Tuple[dict, str, int, int]:
    """Build the explicit graph; returns (workflow, save_node_id, out_w, out_h)."""
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        },
        "4": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["1", 0],
                "preset": "FACEID PLUS V2",
                "lora_strength": float(weights.get("lora", config.FACEID_LORA_STRENGTH)),
                "provider": "CPU",
            },
        },
        "6": {
            "class_type": "IPAdapterFaceID",
            "inputs": {
                "model": ["4", 0],
                "ipadapter": ["4", 1],
                "image": ["2", 0],
                "weight": float(weights.get("weight", config.FACEID_WEIGHT)),
                "weight_faceidv2": float(weights.get("v2", config.FACEID_V2_WEIGHT)),
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.85,
                "embeds_scaling": "K+V w/ C penalty",
            },
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": config.SAMPLER_STEPS,
                "cfg": config.SAMPLER_CFG,
                "sampler_name": config.SAMPLER_NAME,
                "scheduler": config.SCHEDULER,
                "denoise": 1.0,
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["5", 0],
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
    }
    out_w, out_h = int(width), int(height)

    if hires and hires_factor > 1.0:
        workflow["12"] = {
            "class_type": "LatentUpscaleBy",
            "inputs": {"samples": ["9", 0], "upscale_method": "bilinear", "scale_by": float(hires_factor)},
        }
        workflow["13"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed) + 1,
                "steps": int(hires_steps),
                "cfg": config.SAMPLER_CFG,
                "sampler_name": config.SAMPLER_NAME,
                "scheduler": config.SCHEDULER,
                "denoise": float(hires_denoise),
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["12", 0],
            },
        }
        workflow["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}}
        workflow["15"] = {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "Fotogenie_HQ"}}
        out_w = int(round(width * hires_factor))
        out_h = int(round(height * hires_factor))
        save_node = "15"
    else:
        workflow["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "Fotogenie_HQ"}}
        save_node = "11"
    return workflow, save_node, out_w, out_h


def build_detail_workflow(
    reference_name: str,
    crop_name: str,
    positive_prompt: str,
    negative_prompt: str,
    weights: Dict[str, float],
    denoise: float,
    steps: int,
    seed: int,
    checkpoint: str,
) -> Tuple[dict, str]:
    """One high-res pass over an already generated face, FaceID-conditioned.

    The FaceID embedding MUST come from the reference photo (node 2), not from
    the crop being re-rendered, otherwise denoise 0.45 lets the sampler drift
    the face away from the person.
    """
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "3": {"class_type": "LoadImage", "inputs": {"image": crop_name}},
        "4": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["1", 0],
                "preset": "FACEID PLUS V2",
                "lora_strength": float(weights.get("lora", config.FACEID_RETRY_LORA)),
                "provider": "CPU",
            },
        },
        "6": {
            "class_type": "IPAdapterFaceID",
            "inputs": {
                "model": ["4", 0],
                "ipadapter": ["4", 1],
                "image": ["2", 0],
                "weight": float(weights.get("weight", config.FACEID_RETRY_WEIGHT)),
                "weight_faceidv2": float(weights.get("v2", config.FACEID_RETRY_V2_WEIGHT)),
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.85,
                "embeds_scaling": "K+V w/ C penalty",
            },
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": int(steps),
                "cfg": config.SAMPLER_CFG,
                "sampler_name": config.SAMPLER_NAME,
                "scheduler": config.SCHEDULER,
                "denoise": float(denoise),
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["10", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "12": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": "Fotogenie_Detail"},
        },
    }
    return workflow, "12"


def build_upscale_workflow(
    image_name: str, width: int, height: int, model_name: str
) -> Tuple[dict, str]:
    """ESRGAN upscale to an explicit size (ImageScale needs target dims)."""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
        },
        "4": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": int(width),
                "height": int(height),
                "crop": "disabled",
            },
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": "Fotogenie_Final"},
        },
    }
    return workflow, "5"


def build_clothing_inpaint_workflow(
    checkpoint: str,
    person_image_name: str,
    mask_image_name: str,
    face_crop_name: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    weights: Dict[str, float],
    denoise: float = 0.95,
    steps: int = 30,
    grow_mask_by: int = 8,
) -> Tuple[dict, str]:
    """Build SDXL inpainting workflow with FaceID preservation for clothing swap."""
    workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": person_image_name}},
        "3": {
            "class_type": "LoadImageMask",
            "inputs": {"image": mask_image_name, "channel": "red"},
        },
        "4": {"class_type": "LoadImage", "inputs": {"image": face_crop_name}},
        "5": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["1", 0],
                "preset": "FACEID PLUS V2",
                "lora_strength": float(weights.get("lora", config.FACEID_LORA_STRENGTH)),
                "provider": "CPU",
            },
        },
        "6": {
            "class_type": "IPAdapterFaceID",
            "inputs": {
                "model": ["5", 0],
                "ipadapter": ["5", 1],
                "image": ["4", 0],
                "weight": float(weights.get("weight", config.FACEID_WEIGHT)),
                "weight_faceidv2": float(weights.get("v2", config.FACEID_V2_WEIGHT)),
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 0.85,
                "embeds_scaling": "K+V w/ C penalty",
            },
        },
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "9": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2],
                "mask": ["3", 0],
                "grow_mask_by": int(grow_mask_by),
            },
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": int(steps),
                "cfg": config.SAMPLER_CFG,
                "sampler_name": config.SAMPLER_NAME,
                "scheduler": config.SCHEDULER,
                "denoise": float(denoise),
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["9", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
        "12": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": "Fotogenie_DressSwap"},
        },
    }
    return workflow, "12"


def _decode_bgr(image_bytes: bytes):
    if not image_bytes:
        return None
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


def _encode_png(bgr) -> bytes:
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise ComfyGraphError("Could not encode the image.")
    return encoded.tobytes()


def _detail_crop(bgr, bbox, padding: float):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = max(64, int(max(x2 - x1, y2 - y1) * float(padding)))
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(w, left + side)
    bottom = min(h, top + side)
    if right - left < side:
        left = max(0, right - side)
    if bottom - top < side:
        top = max(0, bottom - side)
    return bgr[top:bottom, left:right], (left, top, right, bottom)


def _feather_mask(size: int, feather: float) -> np.ndarray:
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(
        mask,
        (size // 2, size // 2),
        (max(8, int(size * 0.40)), max(8, int(size * 0.46))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    sigma = max(2.0, size * float(feather))
    return cv2.GaussianBlur(mask, (0, 0), sigma)


async def _run_graph(client, workflow: dict, save_node: str, timeout_seconds: int) -> bytes:
    prompt_resp = await client.post(
        f"{config.COMFY_URL}/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex}
    )
    if prompt_resp.status_code != 200:
        raise ComfyGraphError(f"ComfyUI graph error: {prompt_resp.text[:400]}")
    prompt_id = str((prompt_resp.json() or {}).get("prompt_id") or "")
    if not prompt_id:
        raise ComfyGraphError("ComfyUI did not return a prompt id.")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        hist_resp = await client.get(f"{config.COMFY_URL}/history/{prompt_id}")
        if hist_resp.status_code != 200:
            continue
        hist = hist_resp.json() or {}
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id] or {}
        status_obj = entry.get("status") or {}
        if status_obj.get("status_str") == "error":
            message = _error_from_status(status_obj)
            if "out of memory" in message.lower() or "allocation" in message.lower():
                raise OutOfMemory(message)
            raise ComfyGraphError(message)
        outputs = entry.get("outputs") or {}
        images = (outputs.get(save_node) or {}).get("images") or []
        if not images:
            continue
        meta = images[0]
        view = await client.get(
            f"{config.COMFY_URL}/view",
            params={
                "filename": meta.get("filename", ""),
                "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output"),
            },
        )
        if view.status_code != 200 or not view.content:
            raise ComfyGraphError(f"ComfyUI returned no image (HTTP {view.status_code}).")
        return view.content
    raise ComfyTimeout(f"ComfyUI did not finish within {timeout_seconds}s.")


async def run_face_detail(
    image_png: bytes,
    reference_png: bytes,
    positive_prompt: str,
    negative_prompt: str,
    weights: Optional[Dict[str, float]] = None,
    denoise: Optional[float] = None,
    guide_size: Optional[int] = None,
) -> Tuple[bytes, dict]:
    """Re-render the final face at high resolution and composite it back.

    Returns (image_png, meta). On any failure the original image is returned so
    a detail failure never loses an accepted render.
    """
    meta = {"applied": False, "reason": "", "crop": None, "guide": 0, "denoise": 0.0}
    status = await check_comfy_status(force=False)
    if not status.get("online"):
        meta["reason"] = "ComfyUI offline"
        return image_png, meta
    vram = status.get("vram_free_gb")
    if vram is not None and float(vram) < config.DETAIL_MIN_VRAM_GB:
        meta["reason"] = f"low VRAM ({float(vram):.1f} GB)"
        return image_png, meta

    bgr = _decode_bgr(image_png)
    if bgr is None:
        meta["reason"] = "unreadable image"
        return image_png, meta
    try:
        from backend.face_engine import FaceEngine

        face = FaceEngine.get().largest(bgr)
    except Exception as exc:
        meta["reason"] = f"face engine: {exc}"
        return image_png, meta
    if face is None:
        meta["reason"] = "no face"
        return image_png, meta

    checkpoint = status.get("checkpoint")
    if not checkpoint:
        meta["reason"] = "no checkpoint"
        return image_png, meta

    crop, rect = _detail_crop(bgr, face.bbox, config.DETAIL_PADDING)
    if crop.size == 0:
        meta["reason"] = "empty crop"
        return image_png, meta
    left, top, right, bottom = rect
    guide = int(guide_size or config.DETAIL_GUIDE_SIZE)
    crop_size = cv2.resize(crop, (guide, guide), interpolation=cv2.INTER_LANCZOS4)
    mask = _feather_mask(guide, config.DETAIL_FEATHER)
    seed = secrets.randbelow(2 ** 32 - 1) + 1
    weights = weights or {
        "weight": config.FACEID_RETRY_WEIGHT,
        "v2": config.FACEID_RETRY_V2_WEIGHT,
        "lora": config.FACEID_RETRY_LORA,
    }
    detail_denoise = float(config.DETAIL_DENOISE if denoise is None else denoise)

    timeout = httpx.Timeout(connect=5.0, read=45.0, write=45.0, pool=5.0)
    uploaded = ""
    uploaded_crop = ""
    uploaded_ref = ""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            ref_filename = f"fotogenie_detail_ref_{uuid.uuid4().hex}.png"
            ref_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (ref_filename, reference_png, "image/png")},
            )
            if ref_resp.status_code != 200:
                raise ComfyGraphError(
                    f"ComfyUI rejected the reference (HTTP {ref_resp.status_code})."
                )
            uploaded_ref = str((ref_resp.json() or {}).get("name") or ref_filename)

            crop_filename = f"fotogenie_detail_{uuid.uuid4().hex}.png"
            up_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (crop_filename, _encode_png(crop_size), "image/png")},
            )
            if up_resp.status_code != 200:
                raise ComfyGraphError(f"ComfyUI rejected the detail crop (HTTP {up_resp.status_code}).")
            uploaded_crop = str((up_resp.json() or {}).get("name") or crop_filename)

            workflow, save_node = build_detail_workflow(
                reference_name=uploaded_ref,
                crop_name=uploaded_crop,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                weights=weights,
                denoise=detail_denoise,
                steps=config.DETAIL_STEPS,
                seed=seed,
                checkpoint=checkpoint,
            )
            result = await _run_graph(client, workflow, save_node, config.DETAIL_TIMEOUT_SECONDS)
    except Exception as exc:  # A detail failure must never lose an accepted render.
        meta["reason"] = str(exc)[:160]
        logger.warning("face detail skipped: %s", exc)
        return image_png, meta
    finally:
        for name in (uploaded_crop, uploaded_ref):
            if not name:
                continue
            try:
                path = config.COMFY_ROOT / "input" / Path(name).name
                if path.is_file() and path.resolve().parent == (config.COMFY_ROOT / "input").resolve():
                    path.unlink()
            except OSError:
                pass

    rendered = _decode_bgr(result)
    if rendered is None:
        meta["reason"] = "unreadable result"
        return image_png, meta

    region_w, region_h = right - left, bottom - top
    rendered_native = cv2.resize(rendered, (region_w, region_h), interpolation=cv2.INTER_LANCZOS4)
    native_mask = cv2.resize(mask, (region_w, region_h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    out = bgr.copy()
    region = out[top:bottom, left:right].astype(np.float32)
    blended = native_mask * rendered_native.astype(np.float32) + (1.0 - native_mask) * region
    out[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)

    meta.update(
        applied=True,
        reason="",
        crop=[int(left), int(top), int(right), int(bottom)],
        guide=guide,
        denoise=detail_denoise,
    )
    logger.info(
        "face detail applied crop=%s guide=%d denoise=%.2f",
        meta["crop"],
        guide,
        detail_denoise,
    )
    return _encode_png(out), meta


def _upscale_model_path() -> Path:
    return config.COMFY_ROOT / "models" / "upscale_models" / config.UPSCALE_MODEL


async def run_final_upscale(image_png: bytes) -> Tuple[bytes, dict]:
    """ESRGAN upscale to FINAL_UPSCALE. Returns the input on any failure."""
    meta = {"applied": False, "reason": "", "scale": config.FINAL_UPSCALE}
    if not config.ENABLE_FINAL_UPSCALE:
        meta["reason"] = "disabled"
        return image_png, meta
    if not _upscale_model_path().is_file():
        meta["reason"] = f"missing {config.UPSCALE_MODEL}"
        return image_png, meta
    status = await check_comfy_status(force=False)
    if not status.get("online"):
        meta["reason"] = "ComfyUI offline"
        return image_png, meta
    vram = status.get("vram_free_gb")
    if vram is not None and float(vram) < config.UPSCALE_MIN_VRAM_GB:
        meta["reason"] = f"low VRAM ({float(vram):.1f} GB)"
        return image_png, meta
    bgr = _decode_bgr(image_png)
    if bgr is None:
        meta["reason"] = "unreadable image"
        return image_png, meta
    h, w = bgr.shape[:2]
    scale = float(config.FINAL_UPSCALE)
    target_w = max(64, int(round(w * scale)))
    target_h = max(64, int(round(h * scale)))

    timeout = httpx.Timeout(connect=5.0, read=45.0, write=45.0, pool=5.0)
    uploaded = ""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            filename = f"fotogenie_upscale_{uuid.uuid4().hex}.png"
            up_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (filename, image_png, "image/png")},
            )
            if up_resp.status_code != 200:
                raise ComfyGraphError(f"ComfyUI rejected the upscale input (HTTP {up_resp.status_code}).")
            uploaded = str((up_resp.json() or {}).get("name") or filename)
            workflow, save_node = build_upscale_workflow(
                uploaded, target_w, target_h, config.UPSCALE_MODEL
            )
            result = await _run_graph(client, workflow, save_node, config.UPSCALE_TIMEOUT_SECONDS)
    except (ComfyError, OSError, httpx.HTTPError) as exc:
        meta["reason"] = str(exc)[:160]
        logger.warning("final upscale skipped: %s", exc)
        return image_png, meta
    finally:
        if uploaded:
            try:
                path = config.COMFY_ROOT / "input" / Path(uploaded).name
                if path.is_file() and path.resolve().parent == (config.COMFY_ROOT / "input").resolve():
                    path.unlink()
            except OSError:
                pass

    meta.update(applied=True, reason="")
    logger.info("final upscale %dx%d -> %dx%d", w, h, target_w, target_h)
    return result, meta


def _error_from_status(status: dict) -> str:
    for item in status.get("messages") or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == "execution_error":
            info = item[1] or {}
            node = info.get("node_type") or info.get("node_id") or ""
            msg = info.get("exception_message") or info.get("exception_type") or "execution error"
            return f"{node}: {msg}"[:400]
    text = str(status.get("status_str") or "ComfyUI execution error")
    return text[:400]


async def _interrupt(client: httpx.AsyncClient) -> None:
    try:
        await client.post(f"{config.COMFY_URL}/interrupt", json={})
    except Exception:
        pass


async def run_comfy_generation(
    face_crop_png: bytes,
    positive_prompt: str,
    negative_prompt: str,
    aspect: str = "4:5",
    seed: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    hires: Optional[bool] = None,
) -> GenerationResult:
    if not face_crop_png:
        raise ComfyGraphError("The prepared face crop is empty.")
    width, height = _aspect_size(aspect)

    status = await check_comfy_status(force=True)
    if not status["online"]:
        raise ComfyUnavailable(status.get("error") or "ComfyUI is not reachable.")
    if not status["faceid"]:
        raise FaceIdUnavailable(
            "IP-Adapter FaceID is required for identity lock and is not installed in ComfyUI."
        )
    if _missing_models():
        raise FaceIdUnavailable(status.get("error") or "IP-Adapter FaceID model files are missing.")
    checkpoint = status["checkpoint"]
    if not checkpoint:
        raise ComfyGraphError(status.get("error") or "No usable checkpoint.")

    hires_enabled, factor, out_w, out_h = _hires_plan(width, height)
    if hires is not None:
        hires_enabled = bool(hires) and hires_enabled
    if hires_enabled and status.get("vram_free_gb") is not None:
        if float(status["vram_free_gb"]) < config.HIRES_MIN_VRAM_GB:
            logger.warning("Skipping hires pass: only %.2f GB VRAM free", float(status["vram_free_gb"]))
            hires_enabled = False
    if not hires_enabled:
        out_w, out_h = width, height

    chosen_seed = int(seed) if seed is not None else secrets.randbelow(2 ** 32 - 1) + 1
    weight_map = {
        "weight": float((weights or {}).get("weight", config.FACEID_WEIGHT)),
        "v2": float((weights or {}).get("v2", config.FACEID_V2_WEIGHT)),
        "lora": float((weights or {}).get("lora", config.FACEID_LORA_STRENGTH)),
    }

    workflow, save_node, out_w, out_h = build_workflow(
        checkpoint=checkpoint,
        image_name="",
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=chosen_seed,
        weights=weight_map,
        hires=hires_enabled,
        hires_factor=factor,
        hires_steps=config.HIRES_STEPS,
        hires_denoise=config.HIRES_DENOISE,
    )

    timeout = httpx.Timeout(connect=5.0, read=45.0, write=45.0, pool=5.0)
    started = time.monotonic()
    uploaded_name = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            filename = f"fotogenie_ref_{uuid.uuid4().hex}.png"
            up_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (filename, face_crop_png, "image/png")},
            )
            if up_resp.status_code != 200:
                raise ComfyGraphError(f"ComfyUI rejected the reference image (HTTP {up_resp.status_code}).")
            uploaded_name = str((up_resp.json() or {}).get("name") or filename)
            workflow["2"]["inputs"]["image"] = uploaded_name

            prompt_resp = await client.post(
                f"{config.COMFY_URL}/prompt",
                json={"prompt": workflow, "client_id": uuid.uuid4().hex},
            )
            if prompt_resp.status_code != 200:
                text = prompt_resp.text
                raise ComfyGraphError(f"ComfyUI graph error: {text[:400]}")
            prompt_id = str((prompt_resp.json() or {}).get("prompt_id") or "")
            if not prompt_id:
                raise ComfyGraphError("ComfyUI did not return a prompt id.")

            deadline = time.monotonic() + config.COMFY_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                hist_resp = await client.get(f"{config.COMFY_URL}/history/{prompt_id}")
                if hist_resp.status_code != 200:
                    continue
                hist = hist_resp.json() or {}
                if prompt_id not in hist:
                    continue
                entry = hist[prompt_id] or {}
                status_obj = entry.get("status") or {}
                if status_obj.get("status_str") == "error":
                    message = _error_from_status(status_obj)
                    if "out of memory" in message.lower() or "allocation" in message.lower():
                        raise OutOfMemory(message)
                    raise ComfyGraphError(message)
                outputs = entry.get("outputs") or {}
                images = (outputs.get(save_node) or {}).get("images") or []
                if not images:
                    continue
                meta = images[0]
                view = await client.get(
                    f"{config.COMFY_URL}/view",
                    params={
                        "filename": meta.get("filename", ""),
                        "subfolder": meta.get("subfolder", ""),
                        "type": meta.get("type", "output"),
                    },
                )
                if view.status_code != 200 or not view.content:
                    raise ComfyGraphError(f"ComfyUI returned no image (HTTP {view.status_code}).")
                try:
                    with Image.open(BytesIO(view.content)) as probe:
                        actual_w, actual_h = probe.size
                        probe.verify()
                except Exception as exc:
                    raise ComfyGraphError(f"ComfyUI returned an invalid image: {exc}") from exc
                if abs(actual_w - out_w) > 8 or abs(actual_h - out_h) > 8:
                    logger.warning(
                        "Expected %sx%s but ComfyUI returned %sx%s", out_w, out_h, actual_w, actual_h
                    )
                elapsed = time.monotonic() - started
                logger.info(
                    "generated seed=%s ckpt=%s %sx%s hires=%s elapsed=%.1fs",
                    chosen_seed,
                    checkpoint,
                    actual_w,
                    actual_h,
                    hires_enabled,
                    elapsed,
                )
                return GenerationResult(
                    image_bytes=view.content,
                    seed=chosen_seed,
                    checkpoint=checkpoint,
                    width=actual_w,
                    height=actual_h,
                    elapsed=elapsed,
                    face_weights=weight_map,
                    hires=bool(hires_enabled),
                )

            await _interrupt(client)
            raise ComfyTimeout(f"ComfyUI did not finish within {config.COMFY_TIMEOUT_SECONDS}s.")
        except httpx.HTTPError as exc:
            # A dropped connection should read as "engine unavailable", not as an
            # opaque 500 from the generic handler.
            raise ComfyUnavailable(
                f"Lost the connection to ComfyUI while generating: {exc}"
            ) from exc
        finally:
            if uploaded_name:
                try:
                    path = config.COMFY_ROOT / "input" / Path(uploaded_name).name
                    if path.is_file() and path.resolve().parent == (config.COMFY_ROOT / "input").resolve():
                        path.unlink()
                except OSError:
                    pass


async def run_clothing_inpaint(
    image_bytes: bytes,
    mask_bytes: bytes,
    face_crop_png: bytes,
    positive_prompt: str,
    negative_prompt: str,
    seed: Optional[int] = None,
    denoise: float = 0.95,
    steps: int = 30,
    weights: Optional[Dict[str, float]] = None,
) -> GenerationResult:
    """Execute SDXL inpainting with clothing mask and IP-Adapter FaceID identity lock."""
    if not image_bytes:
        raise ComfyGraphError("The input person image is empty.")
    if not mask_bytes:
        raise ComfyGraphError("The clothing mask is empty.")
    if not face_crop_png:
        raise ComfyGraphError("The face reference crop is empty.")

    status = await check_comfy_status(force=True)
    if not status["online"]:
        raise ComfyUnavailable(status.get("error") or "ComfyUI is not reachable.")
    if not status["faceid"]:
        raise FaceIdUnavailable(
            "IP-Adapter FaceID is required for identity lock and is not installed in ComfyUI."
        )
    if _missing_models():
        raise FaceIdUnavailable(status.get("error") or "IP-Adapter FaceID model files are missing.")
    checkpoint = status["checkpoint"]
    if not checkpoint:
        raise ComfyGraphError(status.get("error") or "No usable checkpoint.")

    chosen_seed = int(seed) if seed is not None else secrets.randbelow(2 ** 32 - 1) + 1
    weight_map = {
        "weight": float((weights or {}).get("weight", config.FACEID_WEIGHT)),
        "v2": float((weights or {}).get("v2", config.FACEID_V2_WEIGHT)),
        "lora": float((weights or {}).get("lora", config.FACEID_LORA_STRENGTH)),
    }

    uploaded_person = ""
    uploaded_mask = ""
    uploaded_face = ""
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # 1. Upload person image
            p_filename = f"fotogenie_inpaint_person_{uuid.uuid4().hex}.png"
            p_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (p_filename, image_bytes, "image/png")},
            )
            if p_resp.status_code != 200:
                raise ComfyGraphError(f"ComfyUI rejected person image (HTTP {p_resp.status_code}).")
            uploaded_person = str((p_resp.json() or {}).get("name") or p_filename)

            # 2. Upload mask image
            m_filename = f"fotogenie_inpaint_mask_{uuid.uuid4().hex}.png"
            m_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (m_filename, mask_bytes, "image/png")},
            )
            if m_resp.status_code != 200:
                raise ComfyGraphError(f"ComfyUI rejected mask image (HTTP {m_resp.status_code}).")
            uploaded_mask = str((m_resp.json() or {}).get("name") or m_filename)

            # 3. Upload face reference crop
            f_filename = f"fotogenie_inpaint_face_{uuid.uuid4().hex}.png"
            f_resp = await client.post(
                f"{config.COMFY_URL}/upload/image",
                files={"image": (f_filename, face_crop_png, "image/png")},
            )
            if f_resp.status_code != 200:
                raise ComfyGraphError(f"ComfyUI rejected face crop (HTTP {f_resp.status_code}).")
            uploaded_face = str((f_resp.json() or {}).get("name") or f_filename)

            workflow, save_node = build_clothing_inpaint_workflow(
                checkpoint=checkpoint,
                person_image_name=uploaded_person,
                mask_image_name=uploaded_mask,
                face_crop_name=uploaded_face,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                seed=chosen_seed,
                weights=weight_map,
                denoise=denoise,
                steps=steps,
            )

            result_bytes = await _run_graph(client, workflow, save_node, config.COMFY_TIMEOUT_SECONDS)

            try:
                with Image.open(BytesIO(result_bytes)) as probe:
                    actual_w, actual_h = probe.size
                    probe.verify()
            except Exception as exc:
                raise ComfyGraphError(f"ComfyUI returned an invalid image: {exc}") from exc

            elapsed = time.monotonic() - started
            logger.info(
                "clothing inpaint completed seed=%s ckpt=%s %sx%s elapsed=%.1fs",
                chosen_seed,
                checkpoint,
                actual_w,
                actual_h,
                elapsed,
            )
            return GenerationResult(
                image_bytes=result_bytes,
                seed=chosen_seed,
                checkpoint=checkpoint,
                width=actual_w,
                height=actual_h,
                elapsed=elapsed,
                face_weights=weight_map,
                hires=False,
            )
        except httpx.HTTPError as exc:
            raise ComfyUnavailable(
                f"Lost connection to ComfyUI while inpainting clothing: {exc}"
            ) from exc
        finally:
            for name in (uploaded_person, uploaded_mask, uploaded_face):
                if not name:
                    continue
                try:
                    path = config.COMFY_ROOT / "input" / Path(name).name
                    if path.is_file() and path.resolve().parent == (config.COMFY_ROOT / "input").resolve():
                        path.unlink()
                except OSError:
                    pass
