"""Post-generation identity and face-quality verification on the CPU."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from backend import config
from backend.analyzer import measure_face_skin, measure_face_skin_balanced
from backend.face_engine import FaceEngine
from backend.logging import get_logger

logger = get_logger("fotogenie.face_guard")


def _decode(image_bytes: bytes):
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _normalized(embedding) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    vec = np.asarray(embedding, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def _texture(bgr: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad_x = int(0.12 * max(1, x2 - x1))
    pad_y = int(0.12 * max(1, y2 - y1))
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_metrics(image_bytes: bytes) -> dict:
    bgr = _decode(image_bytes)
    outcome = {
        "face_detected": False,
        "face_count": 0,
        "det_score": 0.0,
        "bbox": None,
        "embedding": None,
        "texture": 0.0,
        "skin_lab": None,
        "skin_lab_balanced": None,
        "width": 0,
        "height": 0,
    }
    if bgr is None:
        return outcome
    outcome["height"], outcome["width"] = bgr.shape[:2]
    try:
        engine = FaceEngine.get()
        faces = engine.detect(bgr)
    except Exception as exc:
        logger.warning("face guard could not load the engine: %s", exc)
        return outcome
    outcome["face_count"] = len(faces)
    if not faces:
        return outcome
    face = faces[0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    outcome.update(
        face_detected=True,
        det_score=float(face.det_score),
        bbox=[int(v) for v in face.bbox],
        embedding=_normalized(getattr(face, "embedding", None)),
        texture=_texture(bgr, face.bbox),
        skin_lab=measure_face_skin(rgb, face),
        skin_lab_balanced=measure_face_skin_balanced(rgb, face),
    )
    return outcome


def _delta_e(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def verify(
    reference_face_crop_png: bytes,
    generated_png: bytes,
    reference_lab: Optional[tuple] = None,
) -> dict:
    ref = face_metrics(reference_face_crop_png)
    gen = face_metrics(generated_png)
    result = {
        "ok": False,
        "identity_similarity": 0.0,
        "face_detected": bool(gen["face_detected"]),
        "face_count": int(gen["face_count"]),
        "det_score": float(gen["det_score"]),
        "texture": float(gen["texture"]),
        "reference_texture": float(ref["texture"]),
        "skin_delta_e": None,
        "skin_delta_chroma": None,
        "reason": "",
    }
    if not ref["face_detected"]:
        result["reason"] = "No face in the reference crop."
        return result
    if not gen["face_detected"]:
        result["reason"] = "No face in the generated image."
        return result
    ref_embedding = ref["embedding"]
    gen_embedding = gen["embedding"]
    if ref_embedding is None or gen_embedding is None:
        result["reason"] = "Face embedding is unavailable."
        return result

    sim = float(np.dot(ref_embedding, gen_embedding))
    ref_lab = reference_lab or ref["skin_lab"]
    gen_lab = gen["skin_lab"]
    # 3D delta-E is context only: L* legitimately changes with scene lighting.
    delta_e = _delta_e(ref_lab, gen_lab)

    # The chroma gate compares cast-normalized undertones on BOTH sides. The
    # reference target (profile.skin_lab) already carries white-balanced a*/b*;
    # measuring the generated side raw counted the scene cast as a skin
    # mismatch and inflated the delta.
    ref_chroma = reference_lab if reference_lab is not None else ref["skin_lab_balanced"]
    gen_chroma = gen["skin_lab_balanced"] or gen["skin_lab"]
    delta_chroma = None
    if ref_chroma is not None and gen_chroma is not None:
        delta_chroma = float(
            math.sqrt(
                (float(ref_chroma[1]) - float(gen_chroma[1])) ** 2
                + (float(ref_chroma[2]) - float(gen_chroma[2])) ** 2
            )
        )
    result.update(
        identity_similarity=round(sim, 4),
        skin_delta_e=round(delta_e, 2) if delta_e is not None else None,
        skin_delta_chroma=round(delta_chroma, 2) if delta_chroma is not None else None,
    )
    result["ok"] = (
        sim >= config.MIN_IDENTITY_SIM
        and int(gen["face_count"]) >= 1
        and float(gen["det_score"]) >= 0.6
    )
    if not result["ok"]:
        result["reason"] = (
            f"Identity similarity {sim:.3f} is below the {config.MIN_IDENTITY_SIM:.2f} gate."
        )
    logger.info(
        "guard sim=%.3f det=%.2f faces=%d texture=%.1f deltaE=%s chroma=%s",
        sim,
        float(gen["det_score"]),
        int(gen["face_count"]),
        float(gen["texture"]),
        result["skin_delta_e"],
        result["skin_delta_chroma"],
    )
    return result


def identity_ok(metrics: dict, min_sim: Optional[float] = None) -> bool:
    """Identity-only acceptance used by the rescue path.

    The rescue must clear RESCUE_MIN_SIM (0.90 by default, calibrated in the
    README) instead of the weaker first-pass sanity floor.
    """
    need = config.RESCUE_MIN_SIM if min_sim is None else float(min_sim)
    return (
        float(metrics.get("identity_similarity") or 0.0) >= need
        and int(metrics.get("face_count") or 0) >= 1
        and float(metrics.get("det_score") or 0.0) >= 0.6
    )


def quality_ok(
    metrics: dict,
    reference_texture: Optional[float] = None,
    min_sim: Optional[float] = None,
    max_chroma: Optional[float] = None,
) -> bool:
    """Identity, a blank-face texture floor, and the skin-tone chroma gate.

    The reference texture is only used for reporting. A face swap restores a
    128px source and is always softer than a native diffusion face, so only a
    truly blank face is rejected. Chroma (a*/b*) is lighting-independent and is
    therefore the honest gate for "the skin tone matches".
    """
    if not identity_ok(metrics, min_sim):
        return False
    texture = float(metrics.get("texture") or 0.0)
    if texture < config.MIN_RESCUE_TEXTURE or texture > config.MAX_FACE_TEXTURE:
        return False
    chroma = metrics.get("skin_delta_chroma")
    limit = config.MAX_SKIN_DELTA_CHROMA if max_chroma is None else float(max_chroma)
    if chroma is not None and float(chroma) > limit:
        return False
    return True
