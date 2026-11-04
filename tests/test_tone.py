from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

from backend import config
from backend.tone import grade_render


def _gray() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _lab_shift(image_rgb: np.ndarray, box, d_a: float, d_b: float) -> np.ndarray:
    """Warm only the face rectangle so whole-image white balance cannot undo it."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(lab.shape[1], x2), min(lab.shape[0], y2)
    lab[y1:y2, x1:x2, 1] = np.clip(lab[y1:y2, x1:x2, 1] + d_a, 0, 255)
    lab[y1:y2, x1:x2, 2] = np.clip(lab[y1:y2, x1:x2, 2] + d_b, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _encode_bgr(bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    return encoded.tobytes()


def _balanced_skin(image_bytes: bytes):
    from backend.analyzer import measure_face_skin_balanced
    from backend.face_engine import FaceEngine

    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    face = FaceEngine.get().largest(bgr)
    return measure_face_skin_balanced(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), face)


def _balanced_delta(image_bytes: bytes, target_lab) -> float:
    current = _balanced_skin(image_bytes)
    return math.hypot(current[1] - target_lab[1], current[2] - target_lab[2])


def test_grade_moves_face_chroma_toward_the_target(preset_path, face_profile):
    src = preset_path.read_bytes()
    target = (
        float(face_profile.skin_lab[0]),
        float(face_profile.skin_lab[1]) + 6.0,
        float(face_profile.skin_lab[2]) - 6.0,
    )
    out = grade_render(src, target)
    before = _balanced_skin(src)
    after = _balanced_skin(out)
    assert before is not None and after is not None
    distance_before = abs(before[1] - target[1]) + abs(before[2] - target[2])
    distance_after = abs(after[1] - target[1]) + abs(after[2] - target[2])
    assert distance_after < distance_before


def test_grade_reaches_the_balanced_target(preset_path, face_profile):
    """A face whose undertone drifted must come back under the chroma gate."""
    from backend.analyzer import load_reference

    src = preset_path.read_bytes()
    target = face_profile.skin_lab
    warmed = _encode_bgr(
        cv2.cvtColor(
            _lab_shift(load_reference(src), face_profile.face_box, 14.0, 12.0),
            cv2.COLOR_RGB2BGR,
        )
    )

    before = _balanced_delta(warmed, target)
    assert before > 6.0  # the fixture really moved the face off-target

    graded = grade_render(warmed, target)
    after = _balanced_delta(graded, target)

    assert after < before
    assert after < config.MAX_SKIN_DELTA_CHROMA


def test_balanced_measurement_is_cast_invariant(face_profile):
    """The gate's measurement removes a uniform cast; the raw one does not."""
    from backend.analyzer import measure_face_skin, measure_face_skin_balanced
    from backend.face_engine import FaceEngine

    bgr = cv2.imdecode(np.frombuffer(face_profile.face_crop_png, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face = FaceEngine.get().largest(bgr)
    base_raw = measure_face_skin(rgb, face)
    base_bal = measure_face_skin_balanced(rgb, face)

    cast = _lab_shift(rgb, [0, 0, rgb.shape[1], rgb.shape[0]], 12.0, 10.0)
    cast_bgr = cv2.cvtColor(cast, cv2.COLOR_RGB2BGR)
    cast_face = FaceEngine.get().largest(cast_bgr)
    cast_raw = measure_face_skin(cast, cast_face)
    cast_bal = measure_face_skin_balanced(cast, cast_face)

    raw_shift = abs(cast_raw[1] - base_raw[1]) + abs(cast_raw[2] - base_raw[2])
    balanced_shift = abs(cast_bal[1] - base_bal[1]) + abs(cast_bal[2] - base_bal[2])
    assert raw_shift > 10.0
    assert balanced_shift < raw_shift * 0.5


def test_grade_is_a_noop_without_a_target(preset_path):
    src = preset_path.read_bytes()
    assert grade_render(src, None) is src


def test_grade_is_a_noop_without_a_face():
    src = _gray()
    out = grade_render(src, (150.0, 140.0, 135.0))
    assert out is src
