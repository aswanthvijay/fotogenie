"""CPU face restoration shared by the analyzer and the rescue swap.

GFPGAN is loaded once (spandrel on CPU) and used for two jobs:

* enhancing the reference face before FaceID/inswapper, so the identity
  embedding is built from a crisp face instead of a 145px LANZOS blow-up;
* restoring the swapped face before it is blended back into the scene.

The reference enhancement keeps the measured skin chroma, so the tone the
analyzer sampled is not shifted by the model.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np

from backend import config
from backend.logging import get_logger

logger = get_logger("fotogenie.restorer")

RESTORE_SIDE = 512


class RestorerUnavailable(RuntimeError):
    """GFPGAN is not installed or could not be loaded."""


def _unsharp(bgr: np.ndarray, amount: float = 0.25, radius: float = 1.0) -> np.ndarray:
    soft = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1.0 + amount, soft, -amount, 0)


def _ellipse_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (size // 2, size // 2),
        (int(size * 0.39), int(size * 0.47)),
        0,
        0,
        360,
        255,
        -1,
    )
    kernel = max(3, int(size * 0.08) | 1)
    return cv2.GaussianBlur(mask, (kernel, kernel), 0)


def _median_lab(bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = (mask > 0) & (lab[..., 0] > 35) & (lab[..., 0] < 252)
    if np.count_nonzero(valid) < 80:
        return None
    return np.median(lab[valid], axis=0)


class FaceRestorer:
    """Lazy, thread-safe GFPGAN wrapper on the CPU."""

    _instance: "Optional[FaceRestorer]" = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "FaceRestorer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if not config.GFPGAN_PATH.is_file():
            raise RestorerUnavailable(f"GFPGAN model is missing at {config.GFPGAN_PATH}")
        try:
            import spandrel
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RestorerUnavailable(f"spandrel/torch are unavailable: {exc}") from exc
        try:
            descriptor = spandrel.ModelLoader().load_from_file(str(config.GFPGAN_PATH))
            self._model = descriptor.model.to("cpu").eval()
            self._torch = torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RestorerUnavailable(f"GFPGAN could not be loaded: {exc}") from exc
        self._face_align = None
        logger.info("GFPGAN restorer ready on CPU")

    @property
    def available(self) -> bool:
        return self._model is not None

    def _align(self):
        if self._face_align is None:
            from insightface.utils import face_align

            self._face_align = face_align
        return self._face_align

    def restore(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Restore an aligned face crop. Input and output are BGR uint8."""
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with self._torch.no_grad():
            output = self._model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        array = output.squeeze(0).permute(1, 2, 0).clamp(-1.0, 1.0).numpy()
        array = ((array + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _restore_in_place(bgr: np.ndarray, restorer: FaceRestorer, face) -> Optional[np.ndarray]:
    face_align = restorer._align()
    aimg, matrix = face_align.norm_crop2(bgr, face.kps, RESTORE_SIDE)
    restored = restorer.restore(aimg)
    if restored.shape[:2] != (RESTORE_SIDE, RESTORE_SIDE):
        restored = cv2.resize(restored, (RESTORE_SIDE, RESTORE_SIDE), interpolation=cv2.INTER_LINEAR)

    mask = _ellipse_mask(RESTORE_SIDE)
    src = _median_lab(aimg, mask)
    dst = _median_lab(restored, mask)
    if src is not None and dst is not None:
        shifted = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB).astype(np.float32)
        alpha = mask.astype(np.float32) / 255.0
        d_l = float(np.clip((src[0] - dst[0]) * 0.35, -6.0, 6.0))
        d_a = float(np.clip((src[1] - dst[1]) * 0.7, -8.0, 8.0))
        d_b = float(np.clip((src[2] - dst[2]) * 0.7, -8.0, 8.0))
        shifted[..., 0] = np.clip(shifted[..., 0] + d_l * alpha, 0, 255)
        shifted[..., 1] = np.clip(shifted[..., 1] + d_a * alpha, 0, 255)
        shifted[..., 2] = np.clip(shifted[..., 2] + d_b * alpha, 0, 255)
        restored = cv2.cvtColor(shifted.astype(np.uint8), cv2.COLOR_LAB2BGR)

    restored = _unsharp(restored, amount=0.35, radius=1.0)
    inverse = cv2.invertAffineTransform(matrix)
    h, w = bgr.shape[:2]
    warped = cv2.warpAffine(restored, inverse, (w, h), borderValue=0)
    warped_mask = cv2.warpAffine(mask, inverse, (w, h)).astype(np.float32)[:, :, None] / 255.0
    merged = warped_mask * warped.astype(np.float32) + (1.0 - warped_mask) * bgr.astype(np.float32)
    return merged.astype(np.uint8)


def enhance_face_crop(png_bytes: bytes) -> Tuple[bytes, bool]:
    """Enhance the face inside an already prepared reference crop.

    Returns (png_bytes, enhanced). Any failure returns the original bytes so a
    missing or broken GFPGAN never blocks a generation.
    """
    if not png_bytes:
        return png_bytes, False
    bgr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return png_bytes, False
    try:
        from backend.face_engine import FaceEngine

        face = FaceEngine.get().largest(bgr)
        if face is None:
            return png_bytes, False
        restorer = FaceRestorer.get_instance()
        merged = _restore_in_place(bgr, restorer, face)
        if merged is None:
            return png_bytes, False
        merged = _unsharp(merged, amount=0.2, radius=1.0)
        ok, encoded = cv2.imencode(".png", merged)
        if not ok:
            return png_bytes, False
        logger.info("reference face enhanced for the identity lock")
        return encoded.tobytes(), True
    except Exception as exc:
        logger.warning("reference face enhancement skipped: %s", exc)
        return png_bytes, False
