"""CPU reference analysis for Fotogenie.

Everything here runs on the CPU. The face is detected with the shared
InsightFace buffalo_l engine, the crop is built around that face, and the
skin profile is sampled from the actual face (forehead, cheeks, nose), not
from the center of the photograph. The old center-of-image sampler asked a
full-body photo for its clothing colour, which then leaked into the prompt.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend import config
from backend.face_engine import FaceEngine, FaceEngineUnavailable
from backend.logging import get_logger

logger = get_logger("fotogenie.analyzer")

try:  # Optional iPhone HEIC support; the app still works without it.
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _HEIF_AVAILABLE = False

MAX_ANALYSIS_SIDE = 2048
MIN_ANALYSIS_SIDE = 256
FACE_CROP_SIDE = 640


class InvalidImageError(ValueError):
    """The upload is not a usable image."""


class NoFaceError(ValueError):
    """No usable face was found in the reference photograph."""


@dataclass(frozen=True)
class ReferenceProfile:
    face_crop_png: bytes
    face_box: Tuple[int, int, int, int]
    face_confidence: float
    face_count: int
    skin_lab: Tuple[float, float, float]
    skin_lab_raw: Tuple[float, float, float]
    skin_descriptor: str
    skin_undertone: str
    body_descriptor: str
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    image_size: Tuple[int, int] = (0, 0)
    face_area_ratio: float = 0.0
    face_texture: float = 0.0
    face_crop_raw_png: bytes = b""
    enhanced: bool = False

    def to_public_dict(self) -> dict:
        return {
            "face_detected": True,
            "face_count": int(self.face_count),
            "face_confidence": round(float(self.face_confidence), 3),
            "face_area_ratio": round(float(self.face_area_ratio), 4),
            "skin_tone": self.skin_descriptor,
            "skin_undertone": self.skin_undertone,
            "warnings": list(self.warnings),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
        }


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def load_reference(image_bytes: bytes) -> np.ndarray:
    """Decode, EXIF-rotate and bound the analysis image."""
    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            img = ImageOps.exif_transpose(opened)
            img = _to_rgb(img)
    except Exception as exc:
        hint = ""
        lower = image_bytes[:32].lower()
        if b"ftyp" in lower and not _HEIF_AVAILABLE:
            hint = " HEIC support requires pillow-heif; install it or upload a JPG/PNG."
        raise InvalidImageError(f"Could not read the image.{hint}") from exc

    if img.width < MIN_ANALYSIS_SIDE or img.height < MIN_ANALYSIS_SIDE:
        raise InvalidImageError(
            f"The image is too small ({img.width}x{img.height}); use at least "
            f"{MIN_ANALYSIS_SIDE}x{MIN_ANALYSIS_SIDE}."
        )

    if max(img.size) > MAX_ANALYSIS_SIDE:
        img.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _skin_mask(rgb: np.ndarray, face) -> np.ndarray:
    """Build a skin mask for the forehead, cheeks and nose of a detected face."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)

    mask = np.zeros((h, w), dtype=np.uint8)
    rects = [
        (x1 + int(0.22 * fw), y1 + int(0.14 * fh), x1 + int(0.78 * fw), y1 + int(0.36 * fh)),
        (x1 + int(0.14 * fw), y1 + int(0.44 * fh), x1 + int(0.40 * fw), y1 + int(0.72 * fh)),
        (x1 + int(0.60 * fw), y1 + int(0.44 * fh), x1 + int(0.86 * fw), y1 + int(0.72 * fh)),
        (x1 + int(0.42 * fw), y1 + int(0.38 * fh), x1 + int(0.58 * fw), y1 + int(0.56 * fh)),
    ]
    for rx1, ry1, rx2, ry2 in rects:
        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)

    # Exclude the eyes and the lips so mascara, iris and lipstick do not enter the sample.
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        for idx in (0, 1, 3, 4):
            px, py = int(kps[idx][0]), int(kps[idx][1])
            cv2.circle(mask, (px, py), max(5, int(0.055 * fw)), 0, -1)
        mx = int((kps[3][0] + kps[4][0]) / 2.0)
        my = int((kps[3][1] + kps[4][1]) / 2.0)
        cv2.ellipse(mask, (mx, my), (int(0.18 * fw), int(0.08 * fh)), 0, 0, 360, 0, -1)

    if np.count_nonzero(mask) < 100:
        mask[:] = 0
        cv2.ellipse(
            mask,
            (x1 + fw // 2, y1 + int(0.52 * fh)),
            (int(0.30 * fw), int(0.26 * fh)),
            0,
            0,
            360,
            255,
            -1,
        )
    return mask


def _lab_median(rgb: np.ndarray, mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([28, 128, 72]), np.array([252, 186, 140])) > 0
    valid = (mask > 0) & skin & (lab[..., 0] > 35) & (lab[..., 0] < 252)
    if np.count_nonzero(valid) < 120:
        valid = (mask > 0) & (lab[..., 0] > 35) & (lab[..., 0] < 252)
    if np.count_nonzero(valid) < 40:
        return None

    pixels = lab[valid]
    med = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels[:, 0] - med[0]))
    if mad > 0.5:
        keep = np.abs(pixels[:, 0] - med[0]) < 2.5 * max(mad, 3.0)
        if np.count_nonzero(keep) >= 40:
            med = np.median(pixels[keep], axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def _white_balance(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the reference illuminant cast with a whole-image Shades-of-Gray estimate.

    The estimate must NOT come from the skin pixels themselves: using skin as
    the illuminant drives real skin toward neutral a*=128, which later grades
    the generated face to a grey-green tone (the visible oval the user saw).
    A whole-image estimate keeps the skin's natural chroma.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    valid = (lab[..., 0] > 25) & (lab[..., 0] < 245)
    pixels = rgb[valid].astype(np.float32) / 255.0
    if pixels.shape[0] < 100:
        return rgb
    p = 6.0
    illum = np.power(np.mean(np.power(pixels, p), axis=0), 1.0 / p)
    if np.any(illum <= 1e-3):
        return rgb
    gain = float(np.mean(illum)) / np.maximum(illum, 1e-3)
    corrected = np.clip(rgb.astype(np.float32) * gain.reshape(1, 1, 3), 0, 255)
    return corrected.astype(np.uint8)


def _undertone(a: float, b: float) -> str:
    if a < 136:
        warmth = "cool"
    elif a <= 144:
        warmth = "neutral"
    elif a <= 152:
        warmth = "warm"
    else:
        warmth = "red"
    if b < 130:
        yellow = "olive"
    elif b <= 140:
        yellow = "neutral golden"
    elif b <= 150:
        yellow = "golden"
    else:
        yellow = "yellow"
    return f"{warmth} {yellow} undertone"


def _lightness(L: float) -> str:
    if L >= 195:
        return "fair"
    if L >= 175:
        return "light"
    if L >= 155:
        return "medium-light"
    if L >= 135:
        return "medium brown"
    if L >= 110:
        return "tan brown"
    if L >= 85:
        return "deep brown"
    return "very deep brown"


def _descriptor(lab: Tuple[float, float, float]) -> str:
    L, a, b = lab
    return (
        f"{_lightness(L)} skin with a {_undertone(a, b)}, natural skin texture, "
        "visible pores, neutral white balance"
    )


def _face_texture(rgb: np.ndarray, face) -> float:
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad_x = int(0.12 * max(1, x2 - x1))
    pad_y = int(0.12 * max(1, y2 - y1))
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_crop(rgb: np.ndarray, face) -> bytes:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)
    cx = x1 + fw // 2
    cy = y1 + fh // 2
    side = int(max(fw, fh) * 2.2)
    side = max(128, side)
    left = cx - side // 2
    top = cy - int(side * 0.48)
    right = left + side
    bottom = top + side

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)
    padded = rgb
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        padded = cv2.copyMakeBorder(
            rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = padded[top:bottom, left:right]
    if crop.size == 0:
        raise NoFaceError("The detected face could not be cropped.")
    crop = cv2.resize(crop, (FACE_CROP_SIDE, FACE_CROP_SIDE), interpolation=cv2.INTER_LANCZOS4)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise InvalidImageError("Could not encode the face crop.")
    return encoded.tobytes()


def measure_face_skin(rgb: np.ndarray, face) -> Optional[Tuple[float, float, float]]:
    """Median LAB of the face skin, used by the guard and the rescue."""
    mask = _skin_mask(rgb, face)
    return _lab_median(rgb, mask)


def measure_face_skin_balanced(rgb: np.ndarray, face) -> Optional[Tuple[float, float, float]]:
    """Median LAB of the face skin after whole-image white balance.

    The reference target's a*/b* is measured on a white-balanced image
    (``analyze_reference`` builds ``skin_lab`` from ``balanced_lab``). Measuring
    the generated side the same way lets the guard compare cast-normalized
    undertones instead of counting the generated scene's illuminant as a skin
    mismatch.
    """
    mask = _skin_mask(rgb, face)
    balanced = _white_balance(rgb, mask)
    return _lab_median(balanced, mask)


def analyze_reference(image_bytes: bytes) -> ReferenceProfile:
    rgb = load_reference(image_bytes)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    engine = FaceEngine.get()
    faces = engine.detect(bgr)
    if not faces:
        raise NoFaceError(
            "No face was detected. Use a clear, front-facing photo of the person you want to generate."
        )

    face = faces[0]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = rgb.shape[:2]
    face_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, w * h))
    warnings = []
    if len(faces) > 1:
        warnings.append("Multiple faces detected; using the largest one.")
    if face_area_ratio < 0.04:
        warnings.append("The face is small in the photo; a closer portrait gives a better identity lock.")
    if float(face.det_score) < 0.65:
        warnings.append("Face detection confidence is low; use a sharper, well-lit photo.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < 35:
        warnings.append("The photo looks blurry; use a sharper reference for best results.")

    mask = _skin_mask(rgb, face)
    raw_lab = _lab_median(rgb, mask)
    if raw_lab is None:
        # Extremely unusual, but never silently invent a tone from clothing.
        raw_lab = (150.0, 140.0, 135.0)
        warnings.append("Skin pixels could not be sampled reliably; a neutral tone was used.")

    balanced = _white_balance(rgb, mask)
    balanced_lab = _lab_median(balanced, mask) or raw_lab
    # Neutralise the reference cast in a*/b* only. Re-normalising L* with the
    # Shades-of-Gray gain brightened medium skin into "light" and made the rescue
    # paste a too-bright face; keep the measured luminance.
    skin_lab = (float(raw_lab[0]), float(balanced_lab[1]), float(balanced_lab[2]))
    descriptor = _descriptor(skin_lab)
    undertone = _undertone(skin_lab[1], skin_lab[2])
    face_crop = _face_crop(rgb, face)
    raw_crop = face_crop
    enhanced = False
    if config.ENABLE_FACE_ENHANCE:
        # A crisp reference embedding is what lifts the rescue from ~0.85 to
        # the ~0.95 the user asked for; a 640px LANCOS blow-up is not enough.
        from backend.restorer import enhance_face_crop

        face_crop, enhanced = enhance_face_crop(raw_crop)
        if not enhanced:
            warnings.append(
                "The reference face could not be enhanced; a sharper photo gives a closer match."
            )

    profile = ReferenceProfile(
        face_crop_png=face_crop,
        face_box=(x1, y1, x2, y2),
        face_confidence=float(face.det_score),
        face_count=len(faces),
        skin_lab=tuple(float(v) for v in skin_lab),
        skin_lab_raw=tuple(float(v) for v in raw_lab),
        skin_descriptor=descriptor,
        skin_undertone=undertone,
        body_descriptor=(
            "natural head-to-shoulder proportions, authentic jawline and cheek fullness, "
            "balanced anatomical figure"
        ),
        warnings=tuple(warnings),
        image_size=(int(w), int(h)),
        face_area_ratio=float(face_area_ratio),
        face_texture=_face_texture(rgb, face),
        face_crop_raw_png=raw_crop,
        enhanced=bool(enhanced),
    )
    logger.info(
        "reference face=%d conf=%.2f area=%.3f skin=(%.1f,%.1f,%.1f) undertone=%s",
        len(faces),
        float(face.det_score),
        face_area_ratio,
        skin_lab[0],
        skin_lab[1],
        skin_lab[2],
        undertone,
    )
    return profile


def _slider(sliders: dict, key: str) -> float:
    try:
        value = float(sliders.get(key, 0))
    except (TypeError, ValueError):
        value = 0.0
    if math.isnan(value) or math.isinf(value):
        value = 0.0
    return max(-2.0, min(2.0, value))


CLOTHING_WORDS = (
    "wearing",
    "dress",
    "saree",
    "sari",
    "blazer",
    "suit",
    "shirt",
    "jeans",
    "gown",
    "lehenga",
    "kurti",
    "outfit",
    "skirt",
    "jacket",
    "coat",
    "bikini",
    "lingerie",
    "robe",
)
EXPLICIT_WORDS = ("nude", "naked", "topless", "lingerie", "bikini", "undressed")
FULL_BODY_WORDS = ("full body", "full-body", "head to toe", "head-to-toe", "standing", "walking", "wide shot")

# CyberRealisticXLPlay responds to a photographic quality header. This block is
# what turns the output from "AI portrait" into "studio DSLR photograph".
QUALITY_HEADER = (
    "masterpiece, best quality, ultra realistic, RAW photo, professional DSLR photograph, "
    "full-frame sensor, 85mm f/1.4 lens, shallow depth of field, high detail, sharp focus, "
    "natural film grain, 8k uhd"
)

SKIN_DETAIL_TOKENS = (
    "natural skin texture with visible pores and fine micro-detail, subsurface scattering, "
    "fine skin detail, natural skin imperfections, realistic skin texture, peach fuzz, "
    "realistic eyelashes and eyebrows, sharp symmetrical eyes, clean natural teeth, "
    "realistic fabric weave and folds, soft natural light, neutral white balance"
)


def build_conditioned_prompt(
    user_prompt: str,
    profile: ReferenceProfile,
    sliders: dict,
    aspect: str = "4:5",
    safe_mode: Optional[bool] = None,
) -> dict:
    safe = config.SAFE_MODE if safe_mode is None else bool(safe_mode)
    prompt = (user_prompt or "").strip()
    p_low = prompt.lower()
    tokens = [QUALITY_HEADER]
    if prompt:
        tokens.append(prompt)

    full_body = any(k in p_low for k in FULL_BODY_WORDS)
    has_outfit = any(k in p_low for k in CLOTHING_WORDS)
    explicit = any(k in p_low for k in EXPLICIT_WORDS)

    if full_body:
        tokens.append("full-body wide shot, head to toe, entire body in frame, standing naturally")
    if safe and not has_outfit and not explicit:
        tokens.append("wearing an elegant tailored outfit, fully clothed, modest fashion")

    # Skin tone is derived from the actual face, with the reference cast removed.
    tokens.append(f"({profile.skin_descriptor}:1.1)")

    h_val = _slider(sliders, "height")
    if any(k in p_low for k in ("taller", "tall body", "long legs", "6ft")) or h_val > 0.3:
        tokens.append("(tall stature, elongated body proportions:1.2)")
    elif any(k in p_low for k in ("short", "petite", "compact height")) or h_val < -0.3:
        tokens.append("(petite compact height, shorter stature:1.2)")

    b_val = _slider(sliders, "bust")
    if any(k in p_low for k in ("larger bust", "big bust", "fuller bust", "large breasts", "busty")) or b_val > 0.3:
        boost = min(1.4, 1.15 + abs(b_val) * 0.12)
        tokens.append(f"(fuller bust, curvy chest proportions:{boost:.2f})")
    elif any(k in p_low for k in ("small bust", "flat chest", "petite chest", "smaller bust")) or b_val < -0.3:
        tokens.append("(slender petite bust:1.2)")

    waist = _slider(sliders, "waist")
    if any(k in p_low for k in ("narrow waist", "tiny waist", "slim waist", "hourglass waist")) or waist < -0.3:
        tokens.append("(narrow hourglass waist, defined midsection:1.2)")
    elif any(k in p_low for k in ("wider waist", "thick waist", "broad waist")) or waist > 0.3:
        tokens.append("(natural fuller waist:1.2)")

    hips = _slider(sliders, "hips")
    if any(k in p_low for k in ("wide hips", "curvy hips", "larger hips", "thick thighs", "curvy figure")) or hips > 0.3:
        boost = min(1.4, 1.15 + abs(hips) * 0.12)
        tokens.append(f"(wide curvy hips, full lower body proportions:{boost:.2f})")
    elif any(k in p_low for k in ("narrow hips", "slim hips", "straight hips")) or hips < -0.3:
        tokens.append("(slender straight hips:1.2)")

    build = _slider(sliders, "build")
    if any(k in p_low for k in ("chubby", "plump", "plus size", "curvy build", "full figured", "thick body")) or build > 0.3:
        tokens.append("(full-figured soft natural body build:1.2)")
    elif any(k in p_low for k in ("skinny", "slender", "thin build", "lean athletic")) or build < -0.3:
        tokens.append("(slender lean body build:1.2)")
    else:
        tokens.append(profile.body_descriptor)

    tokens.append(SKIN_DETAIL_TOKENS)
    positive = ", ".join(t for t in tokens if t)

    negative_tokens = [
        "lowres",
        "low quality",
        "jpeg artifacts",
        "compression artifacts",
        "oversharpened",
        "soft focus",
        "doll skin",
        "smooth wax skin",
        "smooth skin",
        "waxy skin",
        "airbrushed skin",
        "fake skin",
        "pasted face",
        "face mask halo",
        "white edge halo",
        "plastic skin",
        "wax skin",
        "airbrushed",
        "doll face",
        "porcelain skin",
        "over-smooth skin",
        "3d render",
        "CGI",
        "cartoon",
        "illustration",
        "painting",
        "bad anatomy",
        "deformed hands",
        "extra fingers",
        "blurry",
        "watermark",
        "text",
        "blue eyeliner",
        "cyan eyeliner",
        "turquoise eyeshadow",
        "neon makeup",
        "heavy makeup",
        "orange skin",
        "yellow cast",
        "fake tan",
        "oversaturated skin",
        "blotchy skin",
        "grey lips",
        "asymmetrical eyes",
        "cross-eyed",
        "fused teeth",
        "blurry teeth",
        "deformed ear",
        "melted earring",
    ]
    if full_body:
        negative_tokens.extend(["close-up", "headshot", "waist-up", "cropped head", "cropped legs", "portrait crop"])
    if safe and not explicit:
        negative_tokens.extend(
            [
                "nude",
                "naked",
                "topless",
                "exposed breasts",
                "lingerie",
                "nsfw",
                "explicit nudity",
                "cleavage emphasis",
            ]
        )
    negative = ", ".join(negative_tokens)

    return {
        "positive_prompt": positive,
        "negative_prompt": negative,
        "framing": "full_body" if full_body else "standard",
        "applied_sliders": {
            "height": h_val,
            "bust": b_val,
            "waist": waist,
            "hips": hips,
            "build": build,
        },
    }
