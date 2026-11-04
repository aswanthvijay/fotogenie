from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRESET = ROOT / "frontend" / "static" / "presets" / "sample_person.jpg"


@pytest.fixture(scope="session")
def preset_path() -> Path:
    if not PRESET.is_file():
        pytest.skip("bundled preset photo is missing")
    return PRESET


@pytest.fixture(scope="session")
def face_profile(preset_path: Path):
    from backend.analyzer import analyze_reference

    return analyze_reference(preset_path.read_bytes())


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _offline_postprocess(monkeypatch, request):
    """Keep the unit/API suite offline and fast.

    The detail pass, the tone grade and the final upscale talk to ComfyUI and
    the GPU, so they are stubbed here. The GPU-marked integration test keeps
    the real implementations.
    """
    if request.node.get_closest_marker("gpu"):
        return

    async def fake_detail(image_png, *args, **kwargs):
        return image_png, {"applied": False, "reason": "test stub"}

    async def fake_upscale(image_png, *args, **kwargs):
        return image_png, {"applied": False, "reason": "test stub"}

    def fake_grade(image_bytes, target_lab):
        return image_bytes

    monkeypatch.setattr("backend.main.run_face_detail", fake_detail)
    monkeypatch.setattr("backend.main.run_final_upscale", fake_upscale)
    monkeypatch.setattr("backend.main.grade_render", fake_grade)
