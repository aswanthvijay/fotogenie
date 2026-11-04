"""Shared CPU InsightFace engine.

The same buffalo_l model is used by the analyzer, the identity guard and the
optional swap rescue. Loading it once keeps detection/embedding results
consistent and avoids three copies of the model in RAM.

MediaPipe 1.2.0 in this environment no longer exposes mp.solutions, so
InsightFace (CPU) is the reliable CPU-only face detector/embedder.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

from backend import config
from backend.logging import get_logger

logger = get_logger("fotogenie.face_engine")


class FaceEngineUnavailable(RuntimeError):
    """Raised when the InsightFace buffalo_l model cannot be loaded."""


class FaceEngine:
    _instance: "Optional[FaceEngine]" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "FaceEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - environment dependent
            raise FaceEngineUnavailable(
                "insightface is not installed; install the Windows wheel from install.ps1"
            ) from exc

        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                root=str(config.INSIGHTFACE_ROOT),
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:  # pragma: no cover - environment dependent
            raise FaceEngineUnavailable(
                "buffalo_l could not be loaded from "
                f"{config.INSIGHTFACE_ROOT / 'models' / 'buffalo_l'}: {exc}"
            ) from exc
        logger.info("InsightFace buffalo_l loaded on CPU from %s", config.INSIGHTFACE_ROOT)

    def detect(self, bgr: np.ndarray) -> List[object]:
        """Return faces sorted by descending bbox area."""
        if bgr is None or bgr.size == 0:
            return []
        faces = self.app.get(bgr)
        faces.sort(
            key=lambda f: (float(f.bbox[2]) - float(f.bbox[0]))
            * (float(f.bbox[3]) - float(f.bbox[1])),
            reverse=True,
        )
        return faces

    def largest(self, bgr: np.ndarray):
        faces = self.detect(bgr)
        return faces[0] if faces else None
