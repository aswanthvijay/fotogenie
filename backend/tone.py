"""Skin-tone grade applied after all rendering.

Grading must run last: the face-detail pass re-renders skin colour, and the CPU
rescue changes it again.

The reference target's a*/b* is white-balanced (see ``analyze_reference``), so
the current skin chroma is measured in that same cast-normalized space. The
correction is applied inside a feathered, face-bbox-anchored mask (never a
colour mask, which would grab a beige wall) and iterated until the measured
chroma reaches the target, so the face matches the reference undertone without
re-tinting the background.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np

from backend import config
from backend.face_engine import FaceEngine
from backend.logging import get_logger

logger = get_logger("fotogenie.tone")


def _face_skin_mask(shape, face):
    """Soft, face-anchored mask covering forehead, cheeks, jaw and neck.

    Anchored on the detected face bbox, not on skin-colour thresholds: a beige
    wall falls inside a colour skin range and caused the oval the old masked
    grade left behind.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)
    cx = x1 + fw // 2
    cy = y1 + fh // 2
    mask = np.zeros((h, w), np.float32)
    # The ellipse reaches a little below the face box to include jaw and neck.
    cv2.ellipse(
        mask,
        (cx, cy + int(fh * 0.12)),
        (max(8, int(fw * 0.78)), max(8, int(fh * 0.98))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    sigma = max(3.0, min(fw, fh) * float(config.SKIN_MASK_FEATHER))
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    if float(mask.max()) <= 0.0:
        return None
    return mask


def _shift_chroma(bgr: np.ndarray, d_a: float, d_b: float, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip(lab[..., 1] + d_a * mask, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + d_b * mask, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def grade_render(image_bytes: bytes, target_lab: tuple) -> bytes:
    """Shift the rendered skin toward the reference undertone.

    Measures the face skin in the same cast-normalized space as the reference
    target and iterates the masked a*/b* correction until the measured chroma
    is within ``SKIN_GRADE_TOLERANCE``. Returns the input unchanged when no face
    is found or the skin is already on target.
    """
    if not image_bytes or target_lab is None:
        return image_bytes
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return image_bytes
    try:
        from backend.analyzer import measure_face_skin, measure_face_skin_balanced

        face = FaceEngine.get().largest(bgr)
        if face is None:
            return image_bytes
        target = np.asarray(target_lab, dtype=np.float32)
        if target.shape[0] < 3:
            return image_bytes
        mask = _face_skin_mask(bgr.shape, face)
        if mask is None:
            return image_bytes

        work = bgr
        applied = False
        for _ in range(max(1, int(config.SKIN_GRADE_ITERATIONS))):
            current = measure_face_skin_balanced(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), face)
            if current is None:
                break
            d_a = float((target[1] - current[1]) * config.SKIN_CHROMA_STRENGTH)
            d_b = float((target[2] - current[2]) * config.SKIN_CHROMA_STRENGTH)
            if math.hypot(d_a, d_b) < float(config.SKIN_GRADE_TOLERANCE):
                break
            work = _shift_chroma(work, d_a, d_b, mask)
            applied = True

        if not applied:
            return image_bytes

        # A small, raw-space L nudge keeps the face in step without touching the
        # chroma gate (which reads a*/b* only).
        raw = measure_face_skin(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), face)
        if raw is not None:
            d_l = float(np.clip((target[0] - raw[0]) * 0.25, -6.0, 6.0))
            if abs(d_l) >= 0.4:
                lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
                lab[..., 0] = np.clip(lab[..., 0] + d_l, 0, 255)
                work = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        ok, encoded = cv2.imencode(".png", work)
        return encoded.tobytes() if ok else image_bytes
    except Exception as exc:  # A grade failure must never lose the image.
        logger.warning("tone grade skipped: %s", exc)
        return image_bytes
