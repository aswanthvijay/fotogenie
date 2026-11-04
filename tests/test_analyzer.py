from __future__ import annotations

import io

import pytest
from PIL import Image

from backend import config
from backend.analyzer import (
    InvalidImageError,
    NoFaceError,
    _descriptor,
    analyze_reference,
    build_conditioned_prompt,
    load_reference,
)


def _gray_png(size=(512, 512)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def test_face_profile_is_face_based(face_profile):
    assert face_profile.face_count >= 1
    assert face_profile.face_confidence > 0.5
    assert face_profile.face_area_ratio > 0.01
    assert len(face_profile.face_crop_png) > 1000
    crop = Image.open(io.BytesIO(face_profile.face_crop_png))
    assert crop.size == (640, 640)
    assert 0 <= face_profile.skin_lab[0] <= 255
    assert "skin" in face_profile.skin_descriptor
    assert face_profile.face_texture > 0
    assert face_profile.warnings == tuple(face_profile.warnings)


def test_no_face_is_rejected():
    with pytest.raises(NoFaceError):
        analyze_reference(_gray_png())


def test_empty_and_tiny_images_are_rejected():
    with pytest.raises(InvalidImageError):
        analyze_reference(b"")
    with pytest.raises(InvalidImageError):
        analyze_reference(_gray_png((100, 100)))


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (300, 500), (120, 90, 80))
    exif = Image.Exif()
    exif[274] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    rgb = load_reference(buf.getvalue())
    # Orientation 6 swaps the displayed width/height.
    assert rgb.shape[:2] == (300, 500)


def test_undertone_descriptors_can_differ():
    warm = _descriptor((150.0, 148.0, 155.0))
    cool = _descriptor((150.0, 139.0, 130.0))
    assert warm != cool
    assert "warm" in warm or "yellow" in warm


def test_prompt_adds_framing_outfit_and_negative_tokens(face_profile):
    result = build_conditioned_prompt(
        "full body scene on a beach",
        face_profile,
        {"height": 0, "bust": 0, "waist": 0, "hips": 0, "build": 0},
        aspect="9:16",
    )
    positive = result["positive_prompt"].lower()
    negative = result["negative_prompt"].lower()
    assert "full-body" in positive
    assert "wearing an elegant tailored outfit" in positive
    assert "close-up" in negative
    assert "nude" in negative
    assert "plastic skin" in negative
    assert face_profile.skin_descriptor.lower() in positive
    assert result["framing"] == "full_body"


def test_prompt_uses_dslr_quality_language(face_profile):
    result = build_conditioned_prompt("studio portrait", face_profile, {})
    positive = result["positive_prompt"].lower()
    negative = result["negative_prompt"].lower()
    assert "raw photo" in positive
    assert "dslr" in positive
    assert "85mm" in positive
    assert "visible pores" in positive
    assert "lowres" in negative
    assert "jpeg artifacts" in negative
    assert "fully clothed" in positive


def test_explicit_prompt_skips_safe_defaults(face_profile):
    result = build_conditioned_prompt(
        "nude beach photoshoot",
        face_profile,
        {},
        aspect="4:5",
    )
    positive = result["positive_prompt"].lower()
    negative = result["negative_prompt"].lower()
    assert "wearing an elegant tailored outfit" not in positive
    assert "nude" not in negative


def test_bad_sliders_do_not_crash(face_profile):
    result = build_conditioned_prompt(
        "studio portrait",
        face_profile,
        {"height": "abc", "bust": None, "waist": 99, "hips": -99, "build": float("nan")},
    )
    assert result["applied_sliders"]["height"] == 0.0
    assert result["applied_sliders"]["bust"] == 0.0
    assert result["applied_sliders"]["waist"] <= 2.0
    assert result["applied_sliders"]["build"] == 0.0
