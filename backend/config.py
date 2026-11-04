"""Central configuration for Fotogenie.

Every tunable lives here so the API, runner, analyzer and tests agree.
Values can be overridden with environment variables; no module should
hardcode D:/Fotogenie paths.
"""

from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_HTML = FRONTEND_DIR / "index.html"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"

COMFY_ROOT = Path(os.getenv("FOTOGENIE_COMFY_ROOT", str(ROOT / "ComfyUI"))).resolve()
COMFY_URL = os.getenv("FOTOGENIE_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
INSIGHTFACE_ROOT = Path(
    os.getenv("FOTOGENIE_INSIGHTFACE_ROOT", str(COMFY_ROOT / "models" / "insightface"))
).resolve()

# CyberRealisticXLPlay is the primary checkpoint because it produces the most
# photographic skin and pores. It is NSFW-prone, so SAFE_MODE stays on by
# default: the prompt always carries a clothed-outfit clause and the negative
# prompt always suppresses nudity unless the user asks for it explicitly.
CHECKPOINT_PREFERENCE = [
    "CyberRealisticXLPlay_V10.0_FP16.safetensors",
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "RealVisXL_V5.0_fp16.safetensors",
]
PREFERRED_CHECKPOINT = os.getenv("FOTOGENIE_CHECKPOINT", "").strip()
ALLOW_NSFW_CHECKPOINT = _bool("FOTOGENIE_ALLOW_NSFW_CHECKPOINT", True)
ENABLE_FACE_ENHANCE = _bool("FOTOGENIE_ENABLE_FACE_ENHANCE", True)

ASPECT_MAP = {
    "4:5": (896, 1152),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
}

# FaceID strength. The old 0.85 / 0.55 / 0.60 values produced ArcFace similarity
# near 0.33; these are tuned to keep identity without pasting selfie lighting.
FACEID_WEIGHT = _float("FOTOGENIE_FACEID_WEIGHT", 1.15)
FACEID_V2_WEIGHT = _float("FOTOGENIE_FACEID_V2_WEIGHT", 0.65)
FACEID_LORA_STRENGTH = _float("FOTOGENIE_FACEID_LORA", 0.80)
FACEID_RETRY_WEIGHT = _float("FOTOGENIE_FACEID_RETRY_WEIGHT", 1.25)
FACEID_RETRY_V2_WEIGHT = _float("FOTOGENIE_FACEID_RETRY_V2_WEIGHT", 0.75)
FACEID_RETRY_LORA = _float("FOTOGENIE_FACEID_RETRY_LORA", 0.95)

SAMPLER_STEPS = _int("FOTOGENIE_STEPS", 32)
SAMPLER_CFG = _float("FOTOGENIE_CFG", 5.0)
SAMPLER_NAME = os.getenv("FOTOGENIE_SAMPLER", "dpmpp_2m_sde")
SCHEDULER = os.getenv("FOTOGENIE_SCHEDULER", "karras")

ENABLE_HIRES = _bool("FOTOGENIE_ENABLE_HIRES", True)
HIRES_FACTOR = _float("FOTOGENIE_HIRES_FACTOR", 1.35)
HIRES_DENOISE = _float("FOTOGENIE_HIRES_DENOISE", 0.42)
HIRES_STEPS = _int("FOTOGENIE_HIRES_STEPS", 18)
MAX_HIRES_PIXELS = _int("FOTOGENIE_MAX_HIRES_PIXELS", 1_800_000)
HIRES_MIN_VRAM_GB = _float("FOTOGENIE_HIRES_MIN_VRAM_GB", 1.5)

# First-pass FaceID acceptance. This is only a sanity floor; the face the user
# actually gets is the enhanced-reference rescue, which must clear RESCUE_MIN_SIM.
MIN_IDENTITY_SIM = _float("FOTOGENIE_MIN_IDENTITY_SIM", 0.45)
# A very low first-pass similarity will not recover enough from a stronger retry;
# go straight to the rescue instead of burning another two minutes on the GPU.
IDENTITY_RETRY_MIN = _float("FOTOGENIE_IDENTITY_RETRY_MIN", 0.40)
# The rescue face starts from a 128px inswapper output and is always softer than
# the diffusion face. Reject only truly blank faces; texture quality is reported,
# not used to dead-end the user.
MIN_RESCUE_TEXTURE = _float("FOTOGENIE_MIN_RESCUE_TEXTURE", 15.0)
MAX_FACE_TEXTURE = _float("FOTOGENIE_MAX_FACE_TEXTURE", 450.0)
MAX_SKIN_DELTA_E = _float("FOTOGENIE_MAX_SKIN_DELTA_E", 8.0)

# The user requirement is a ~95% face match. ArcFace cosine with a small
# reference is calibrated in README.md; the acceptance floor lives here.
RESCUE_MIN_SIM = _float("FOTOGENIE_RESCUE_MIN_SIM", 0.86)
RESCUE_TARGET_SIM = _float("FOTOGENIE_RESCUE_TARGET_SIM", 0.95)
# Below this the rescue cannot plausibly cross the gate; do not spend a second
# GPU pass on it.
RESCUE_RETRY_MIN = _float("FOTOGENIE_RESCUE_RETRY_MIN", 0.80)
RESCUE_MAX_ATTEMPTS = _int("FOTOGENIE_RESCUE_MAX_ATTEMPTS", 2)

# Skin-tone matching. Chroma (a*/b*) is the comparatively lighting-independent
# part of the tone. The reference target a*/b* is white-balanced, so the
# generated side is measured in that same cast-normalized space before it is
# compared or graded.
MAX_SKIN_DELTA_CHROMA = _float("FOTOGENIE_MAX_SKIN_DELTA_CHROMA", 8.0)
# Skin-anchored chroma grade. The correction is applied inside a feathered,
# face-bbox-anchored mask (never a colour mask, which would grab a beige wall)
# and iterated until the measured chroma is within tolerance of the reference.
# This lets the face reach the undertone without re-tinting the background.
SKIN_CHROMA_STRENGTH = _float("FOTOGENIE_SKIN_CHROMA_STRENGTH", 1.0)
SKIN_MASK_FEATHER = _float("FOTOGENIE_SKIN_MASK_FEATHER", 0.25)
SKIN_GRADE_ITERATIONS = _int("FOTOGENIE_SKIN_GRADE_ITERATIONS", 4)
SKIN_GRADE_TOLERANCE = _float("FOTOGENIE_SKIN_GRADE_TOLERANCE", 1.0)
# Measured on the real reference: injecting the generated scene's own
# high-frequency detail into the swapped face is the only texture source that
# does not move identity. At 1.0 the face reaches texture ~103 with a 0.90
# match (was 42 with 0.25).
RESCUE_DETAIL_STRENGTH = _float("FOTOGENIE_RESCUE_DETAIL", 1.0)
SWAP_SHARPEN_AMOUNT = _float("FOTOGENIE_SWAP_SHARPEN", 1.4)

# Face detail pass: re-render the final face at high resolution with the same
# FaceID conditioning, then composite it back. This is what turns the soft
# 128px inswapper face into a photographic one.
# A diffusion re-render of the face was measured and rejected: even at
# denoise 0.25 it drops ArcFace similarity from 0.93 to 0.59, because FaceID is
# too weak a conditioner to hold identity through a re-render. The flag stays
# for experiments; the texture comes from detail injection instead.
ENABLE_FACE_DETAIL = _bool("FOTOGENIE_ENABLE_FACE_DETAIL", False)
DETAIL_GUIDE_SIZE = _int("FOTOGENIE_DETAIL_GUIDE_SIZE", 768)
DETAIL_PADDING = _float("FOTOGENIE_DETAIL_PADDING", 1.6)
DETAIL_DENOISE = _float("FOTOGENIE_DETAIL_DENOISE", 0.45)
DETAIL_STEPS = _int("FOTOGENIE_DETAIL_STEPS", 22)
DETAIL_FEATHER = _float("FOTOGENIE_DETAIL_FEATHER", 0.12)
DETAIL_MIN_VRAM_GB = _float("FOTOGENIE_DETAIL_MIN_VRAM_GB", 1.2)
DETAIL_TIMEOUT_SECONDS = _int("FOTOGENIE_DETAIL_TIMEOUT", 300)

# Final upscale with a photo upscaler model (ESRGAN class).
ENABLE_FINAL_UPSCALE = _bool("FOTOGENIE_ENABLE_FINAL_UPSCALE", True)
FINAL_UPSCALE = _float("FOTOGENIE_FINAL_UPSCALE", 1.25)
UPSCALE_MODEL = os.getenv("FOTOGENIE_UPSCALE_MODEL", "4x-UltraSharp.pth")
UPSCALE_MIN_VRAM_GB = _float("FOTOGENIE_UPSCALE_MIN_VRAM_GB", 1.5)
UPSCALE_TIMEOUT_SECONDS = _int("FOTOGENIE_UPSCALE_TIMEOUT", 240)
FINAL_SHARPEN_AMOUNT = _float("FOTOGENIE_FINAL_SHARPEN", 0.3)

ENABLE_IDENTITY_RESCUE = _bool("FOTOGENIE_ENABLE_IDENTITY_RESCUE", True)
SAFE_MODE = _bool("FOTOGENIE_SAFE_MODE", True)

MAX_UPLOAD_MB = _int("FOTOGENIE_MAX_UPLOAD_MB", 20)
COMFY_TIMEOUT_SECONDS = _int("FOTOGENIE_COMFY_TIMEOUT", 240)
STATUS_CACHE_SECONDS = _float("FOTOGENIE_STATUS_CACHE_SECONDS", 4.0)

INSWAPPER_PATH = Path(
    os.getenv("FOTOGENIE_INSWAPPER", str(Path.home() / ".insightface" / "models" / "inswapper_128.onnx"))
)
GFPGAN_PATH = Path(
    os.getenv("FOTOGENIE_GFPGAN", str(Path.home() / ".insightface" / "models" / "GFPGANv1.4.pth"))
)

LOG_LEVEL = os.getenv("FOTOGENIE_LOG_LEVEL", "INFO").upper()

for _dir in (OUTPUTS_DIR, LOGS_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
