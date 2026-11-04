"""CPU identity-rescue face swap.

This is not the default generation path. It only runs when the ComfyUI FaceID
result fails the identity gate, which is normal for a small reference face.

The swap is inswapper_128. The raw swap is kept because GFPGAN restoration
measurably drifts identity; generated scene detail is re-injected into the
face, and the whole render is then colour-graded globally toward the reference
skin chroma. The grade is global on purpose: an individually masked shift
leaves a visible oval on the background when the scene light is warmer than the
reference (the beige wall falls inside the "skin" range), which is exactly the
patch the user saw. A global grade has no mask edge at all.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np

from backend import config
from backend.face_engine import FaceEngine
from backend.logging import get_logger

logger = get_logger("fotogenie.face_swapper")

ALIGN_SIDE = 512


class RescueUnavailable(RuntimeError):
    pass


def _unsharp(bgr: np.ndarray, amount: float = 0.2, radius: float = 1.0) -> np.ndarray:
    soft = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1.0 + amount, soft, -amount, 0)


def _face_skin_mask(bgr: np.ndarray, face) -> np.ndarray:
    """Forehead / cheeks / nose boxes inside the detected face."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)
    mask = np.zeros((h, w), np.uint8)
    rects = [
        (x1 + int(0.22 * fw), y1 + int(0.14 * fh), x1 + int(0.78 * fw), y1 + int(0.36 * fh)),
        (x1 + int(0.14 * fw), y1 + int(0.44 * fh), x1 + int(0.40 * fw), y1 + int(0.72 * fh)),
        (x1 + int(0.60 * fw), y1 + int(0.44 * fh), x1 + int(0.86 * fw), y1 + int(0.72 * fh)),
        (x1 + int(0.42 * fw), y1 + int(0.38 * fh), x1 + int(0.58 * fw), y1 + int(0.56 * fh)),
    ]
    for rx1, ry1, rx2, ry2 in rects:
        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
    return mask


class FaceSwapper:
    _instance: "Optional[FaceSwapper]" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "FaceSwapper":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if not config.INSWAPPER_PATH.is_file():
            raise RescueUnavailable(f"inswapper model is missing at {config.INSWAPPER_PATH}")
        try:
            from insightface.model_zoo.inswapper import INSwapper
            from insightface.utils import face_align
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RescueUnavailable(f"InsightFace swap modules are unavailable: {exc}") from exc

        self._face_align = face_align
        self.swapper = INSwapper(str(config.INSWAPPER_PATH))
        self.engine = FaceEngine.get()

    def swap_face(
        self,
        gen_image_bytes: bytes,
        ref_image_bytes: bytes,
        target_lab: Optional[tuple] = None,  # kept for the caller's API; grading runs in main
    ) -> bytes:
        if not gen_image_bytes or not ref_image_bytes:
            raise RescueUnavailable("Missing image bytes for the rescue swap.")
        gen = cv2.imdecode(np.frombuffer(gen_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        ref = cv2.imdecode(np.frombuffer(ref_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if gen is None or ref is None:
            raise RescueUnavailable("Could not decode the images for the rescue swap.")

        gen_face = self.engine.largest(gen)
        ref_face = self.engine.largest(ref)
        if gen_face is None:
            raise RescueUnavailable("No face was detected in the generated image.")
        if ref_face is None:
            raise RescueUnavailable("No face was detected in the reference crop.")

        # Align the generated face once so its own high-frequency detail can be
        # re-injected after the swap; the high-pass never comes from the
        # reference, so no copied highlights.
        orig_face_bgr = None
        try:
            orig_face_bgr, _ = self._face_align.norm_crop2(gen, gen_face.kps, ALIGN_SIDE)
        except Exception:
            orig_face_bgr = None

        swapped = self.swapper.get(gen, gen_face, ref_face, paste_back=True)
        if swapped is None:
            raise RescueUnavailable("The inswapper model did not return an image.")

        # GFPGAN is intentionally NOT run on the swapped face: measured on the
        # real reference it drops ArcFace similarity from 0.936 to 0.875.
        try:
            swapped = self._polish_face(swapped, orig_face_bgr)
        except Exception as exc:  # Keep the raw swap rather than failing the rescue.
            logger.warning("face polish skipped: %s", exc)

        # Colour grading happens in main, after the face-detail pass, so the
        # re-rendered skin is graded too (backend/tone.py:grade_render).
        swapped = _unsharp(swapped, amount=config.FINAL_SHARPEN_AMOUNT, radius=0.8)

        ok, encoded = cv2.imencode(".png", swapped)
        if not ok:
            raise RescueUnavailable("Could not encode the rescue output.")
        return encoded.tobytes()

    def _polish_face(
        self,
        bgr: np.ndarray,
        orig_face_bgr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Re-inject the generated scene's micro-detail into the swapped face."""
        face = self.engine.largest(bgr)
        if face is None:
            return bgr
        aimg, matrix = self._face_align.norm_crop2(bgr, face.kps, ALIGN_SIDE)

        mask = np.zeros((ALIGN_SIDE, ALIGN_SIDE), dtype=np.float32)
        cv2.ellipse(mask, (256, 258), (188, 226), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (41, 41), 15.0)

        if orig_face_bgr is not None and orig_face_bgr.shape[:2] == (ALIGN_SIDE, ALIGN_SIDE):
            gray = cv2.cvtColor(orig_face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            smooth = cv2.GaussianBlur(gray, (0, 0), 1.6)
            detail = np.clip(gray - smooth, -40.0, 40.0)
            aimg = np.clip(
                aimg.astype(np.float32)
                + detail[:, :, None] * mask[:, :, None] * config.RESCUE_DETAIL_STRENGTH,
                0,
                255,
            ).astype(np.uint8)

        aimg = _unsharp(aimg, amount=config.SWAP_SHARPEN_AMOUNT, radius=0.8)

        inverse = cv2.invertAffineTransform(matrix)
        h, w = bgr.shape[:2]
        warped = cv2.warpAffine(aimg, inverse, (w, h), borderValue=0.0)
        warped_mask = cv2.warpAffine(mask, inverse, (w, h), borderValue=0.0)[:, :, None]
        return (
            warped_mask * warped.astype(np.float32)
            + (1.0 - warped_mask) * bgr.astype(np.float32)
        ).astype(np.uint8)

# Colour grading lives in backend/tone.py so it can run after the face-detail
# pass, on both the FaceID path and the rescue path.
