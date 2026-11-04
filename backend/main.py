"""FastAPI studio backend.

Sequential generation queue, face-based reference analysis, an identity
gate after generation, prompt-based clothing inpainting, and an optional CPU
rescue only when the gate fails. No text-only fallback and no unverified
wrong-face image is ever returned.
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import time
import traceback
import uuid
from io import BytesIO
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend import config
from backend.analyzer import (
    InvalidImageError,
    NoFaceError,
    ReferenceProfile,
    analyze_reference,
    build_conditioned_prompt,
)
from backend.clothing_parser import ClothingParser
from backend.comfy_runner import (
    ComfyGraphError,
    ComfyTimeout,
    ComfyUnavailable,
    FaceIdUnavailable,
    GenerationResult,
    OutOfMemory,
    check_comfy_status,
    run_clothing_inpaint,
    run_comfy_generation,
    run_face_detail,
    run_final_upscale,
)
from backend.face_engine import FaceEngine, FaceEngineUnavailable
from backend.face_guard import quality_ok, verify
from backend.logging import bind_request_id, configure_logging, get_logger
from backend.queue_manager import JobStatus, QueueManager
from backend.tone import grade_render

configure_logging()
logger = get_logger("fotogenie.api")

GENERATION_LOCK = asyncio.Lock()
FACE_ENGINE_STATE = {"ready": False, "error": ""}

RESCUE_WEIGHTS = {
    "weight": config.FACEID_RETRY_WEIGHT,
    "v2": config.FACEID_RETRY_V2_WEIGHT,
    "lora": config.FACEID_RETRY_LORA,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    queue_mgr = QueueManager.get_instance()
    queue_mgr.start()
    try:
        await asyncio.to_thread(FaceEngine.get)
        FACE_ENGINE_STATE["ready"] = True
        FACE_ENGINE_STATE["error"] = ""
        logger.info("startup: face engine ready")
    except Exception as exc:
        FACE_ENGINE_STATE["ready"] = False
        FACE_ENGINE_STATE["error"] = str(exc)
        logger.warning("startup: face engine unavailable: %s", exc)
    try:
        yield
    finally:
        await queue_mgr.stop()


app = FastAPI(title="Fotogenie Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if (config.FRONTEND_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR / "static")), name="static")


def _request_id(request: Request) -> str:
    return (request.headers.get("x-request-id") or uuid.uuid4().hex)[:32]


async def _read_upload(image: UploadFile) -> bytes:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"The image is larger than {config.MAX_UPLOAD_MB} MB. Please resize it first.",
        )
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG or HEIC image.")
    return raw


def _parse_sliders(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Body sliders are not valid JSON.") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="Body sliders must be an object.")
    return value


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    try:
        return HTMLResponse(config.FRONTEND_HTML.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="The studio UI file is missing.") from exc


@app.get("/api/status")
async def api_status() -> dict:
    status = await check_comfy_status()
    q_stats = QueueManager.get_instance().get_stats()
    status.update(
        busy=q_stats["busy"],
        queue_count=q_stats["queue_count"],
        current_job_id=q_stats["current_job_id"],
        face_engine=FACE_ENGINE_STATE["ready"],
        face_engine_error=FACE_ENGINE_STATE["error"],
    )
    return status


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str) -> dict:
    job = QueueManager.get_instance().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job.to_dict()


@app.post("/api/preflight")
async def api_preflight(image: UploadFile = File(...)) -> dict:
    raw = await _read_upload(image)
    try:
        profile = await asyncio.to_thread(analyze_reference, raw)
    except NoFaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FaceEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return profile.to_public_dict()


async def _execute_generate(
    raw: bytes,
    user_prompt: str,
    aspect: str,
    slider_dict: dict,
    request_id: str,
) -> dict:
    request_log = bind_request_id(logger, request_id)
    started = time.monotonic()

    try:
        profile: ReferenceProfile = await asyncio.to_thread(analyze_reference, raw)
    except NoFaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FaceEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    effective_prompt = (user_prompt or "").strip() or "cinematic portrait photoshoot, soft natural lighting"
    conditioned = build_conditioned_prompt(
        effective_prompt, profile, slider_dict, aspect=aspect, safe_mode=config.SAFE_MODE
    )

    async with GENERATION_LOCK:
        try:
            status = await check_comfy_status(force=True)
            if not status.get("ready"):
                detail = status.get("error") or "ComfyUI is not ready for generation."
                raise HTTPException(status_code=503, detail=detail)

            result = await run_comfy_generation(
                profile.face_crop_png,
                conditioned["positive_prompt"],
                conditioned["negative_prompt"],
                aspect=aspect,
            )
            verification = await asyncio.to_thread(
                verify, profile.face_crop_png, result.image_bytes, profile.skin_lab
            )

            needs_rescue = float(verification["identity_similarity"]) < config.RESCUE_MIN_SIM
            if needs_rescue and not config.ENABLE_IDENTITY_RESCUE:
                if verification["identity_similarity"] >= config.IDENTITY_RETRY_MIN:
                    request_log.info(
                        "identity below delivery bar (%.3f); retrying with stronger FaceID",
                        verification["identity_similarity"],
                    )
                    result = await run_comfy_generation(
                        profile.face_crop_png,
                        conditioned["positive_prompt"],
                        conditioned["negative_prompt"],
                        aspect=aspect,
                        weights=RESCUE_WEIGHTS,
                    )
                    verification = await asyncio.to_thread(
                        verify, profile.face_crop_png, result.image_bytes, profile.skin_lab
                    )
                needs_rescue = float(verification["identity_similarity"]) < config.RESCUE_MIN_SIM

            rescue_applied = False
            rescue_note = ""
            rescue_attempts = 0
            if needs_rescue and config.ENABLE_IDENTITY_RESCUE:
                best = None
                for attempt in range(max(1, config.RESCUE_MAX_ATTEMPTS)):
                    if attempt > 0:
                        request_log.info("rescue attempt %d: regenerating", attempt + 1)
                        result = await run_comfy_generation(
                            profile.face_crop_png,
                            conditioned["positive_prompt"],
                            conditioned["negative_prompt"],
                            aspect=aspect,
                        )
                    rescue_attempts = attempt + 1
                    rescued_bytes, rescue_metrics = await asyncio.to_thread(
                        _run_rescue, result.image_bytes, profile
                    )
                    if rescued_bytes is None or rescue_metrics is None:
                        break
                    if best is None or float(rescue_metrics.get("identity_similarity") or 0) > float(
                        best[1].get("identity_similarity") or 0
                    ):
                        best = (rescued_bytes, rescue_metrics)
                    if rescue_metrics.get("ok") and quality_ok(
                        rescue_metrics,
                        rescue_metrics.get("reference_texture"),
                        min_sim=config.RESCUE_MIN_SIM,
                    ):
                        break
                    if float(rescue_metrics.get("identity_similarity") or 0) < config.RESCUE_RETRY_MIN:
                        break

                if best is not None:
                    rescued_bytes, rescue_metrics = best
                    if rescue_metrics.get("ok") and quality_ok(
                        rescue_metrics,
                        rescue_metrics.get("reference_texture"),
                        min_sim=config.RESCUE_MIN_SIM,
                    ):
                        result = replace(result, image_bytes=rescued_bytes)
                        verification = rescue_metrics
                        rescue_applied = True
                    else:
                        similarity = float(rescue_metrics.get("identity_similarity") or 0)
                        chroma = rescue_metrics.get("skin_delta_chroma")
                        if not rescue_metrics.get("ok"):
                            rescue_note = "The face swap could not preserve the reference; use a clearer photo."
                        elif similarity < config.RESCUE_MIN_SIM:
                            rescue_note = f"The best face match was {similarity:.0%}, below {config.RESCUE_MIN_SIM:.0%}."
                        elif chroma is not None and float(chroma) > config.MAX_SKIN_DELTA_CHROMA:
                            rescue_note = f"Skin tone delta {float(chroma):.1f} exceeded limit."
                        else:
                            rescue_note = "The face swap produced artifacts; try a sharper reference."
                else:
                    rescue_note = "Identity rescue is unavailable on this machine."

            # Final render: detail pass -> global grade -> upscale
            face_detail_applied = False
            upscaled = False
            texture_before = float(verification.get("texture") or 0.0)
            detail_reason = "disabled" if not config.ENABLE_FACE_DETAIL else ""

            if verification["ok"]:
                if rescue_applied:
                    base_bytes = result.image_bytes
                else:
                    base_bytes = await asyncio.to_thread(
                        grade_render, result.image_bytes, profile.skin_lab
                    )
                base_verification = await asyncio.to_thread(
                    verify, profile.face_crop_png, base_bytes, profile.skin_lab
                )
                final_bytes = base_bytes
                final_verification = base_verification
                texture_before = float(base_verification.get("texture") or 0.0)

                if config.ENABLE_FACE_DETAIL:
                    detail_bytes, detail_meta = await run_face_detail(
                        result.image_bytes,
                        profile.face_crop_png,
                        conditioned["positive_prompt"],
                        conditioned["negative_prompt"],
                        weights=RESCUE_WEIGHTS,
                    )
                    detail_reason = str(detail_meta.get("reason") or "")
                    if detail_meta.get("applied"):
                        graded_detail = await asyncio.to_thread(
                            grade_render, detail_bytes, profile.skin_lab
                        )
                        detail_verification = await asyncio.to_thread(
                            verify, profile.face_crop_png, graded_detail, profile.skin_lab
                        )
                        detail_sim = float(detail_verification.get("identity_similarity") or 0)
                        base_sim = float(base_verification.get("identity_similarity") or 0)
                        if detail_verification.get("ok") and detail_sim >= base_sim - 0.03:
                            final_bytes = graded_detail
                            final_verification = detail_verification
                            face_detail_applied = True

                upscaled_bytes, upscale_meta = await run_final_upscale(final_bytes)
                if upscale_meta.get("applied"):
                    upscaled_verification = await asyncio.to_thread(
                        verify, profile.face_crop_png, upscaled_bytes, profile.skin_lab
                    )
                    if upscaled_verification.get("ok"):
                        final_bytes = upscaled_bytes
                        final_verification = upscaled_verification
                        upscaled = True

                try:
                    with Image.open(BytesIO(final_bytes)) as probe:
                        final_w, final_h = probe.size
                except Exception:
                    final_w, final_h = result.width, result.height
                result = replace(
                    result,
                    image_bytes=final_bytes,
                    width=int(final_w),
                    height=int(final_h),
                )
                verification = final_verification

            chroma = verification.get("skin_delta_chroma")
            chroma_failed = chroma is not None and float(chroma) > config.MAX_SKIN_DELTA_CHROMA
            if not verification["ok"] or float(verification["identity_similarity"]) < config.RESCUE_MIN_SIM or chroma_failed:
                if rescue_note:
                    detail = rescue_note
                elif float(verification["identity_similarity"]) < config.RESCUE_MIN_SIM:
                    detail = f"The best face match was {float(verification['identity_similarity']):.0%}, below {config.RESCUE_MIN_SIM:.0%}."
                elif chroma_failed:
                    detail = f"Skin tone delta {float(chroma):.1f} exceeded limit {config.MAX_SKIN_DELTA_CHROMA:.1f}."
                else:
                    detail = verification.get("reason") or "Face did not match reference closely enough."
                raise HTTPException(status_code=422, detail=detail)

            elapsed_ms = int((time.monotonic() - started) * 1000)
            return {
                "image_base64": "data:image/png;base64," + base64.b64encode(result.image_bytes).decode("ascii"),
                "applied_prompt": conditioned["positive_prompt"],
                "seed": int(result.seed),
                "width": int(result.width),
                "height": int(result.height),
                "checkpoint": result.checkpoint,
                "identity_similarity": float(verification["identity_similarity"]),
                "identity_match_percent": round(float(verification["identity_similarity"]) * 100.0, 1),
                "identity_verified": bool(verification["ok"]),
                "skin_delta_e": verification.get("skin_delta_e"),
                "skin_delta_chroma": verification.get("skin_delta_chroma"),
                "skin_match_percent": (
                    round(max(0.0, 100.0 - float(verification["skin_delta_chroma"]) * 5.0), 1)
                    if verification.get("skin_delta_chroma") is not None
                    else None
                ),
                "face_texture": verification.get("texture"),
                "face_enhanced": bool(profile.enhanced),
                "rescue_applied": bool(rescue_applied),
                "rescue_attempts": int(rescue_attempts),
                "face_detail_applied": bool(face_detail_applied),
                "face_detail_reason": detail_reason,
                "face_texture_before": round(float(texture_before), 1),
                "upscaled": bool(upscaled),
                "elapsed_ms": elapsed_ms,
                "request_id": request_id,
            }
        except HTTPException:
            raise
        except FaceIdUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ComfyUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ComfyTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except OutOfMemory as exc:
            raise HTTPException(status_code=503, detail=f"The GPU ran out of memory: {exc}") from exc
        except ComfyGraphError as exc:
            raise HTTPException(status_code=502, detail=f"ComfyUI could not render this scene: {exc}") from exc
        except Exception as exc:
            request_log.error("generation failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail="Unexpected generation error; check the server log.") from exc


async def _execute_dress_swap(
    raw_image: bytes,
    dress_prompt: str,
    category: str,
    garment_ref_bytes: Optional[bytes],
    request_id: str,
) -> dict:
    request_log = bind_request_id(logger, request_id)
    started = time.monotonic()

    try:
        profile: ReferenceProfile = await asyncio.to_thread(analyze_reference, raw_image)
    except NoFaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FaceEngineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Clothing mask extraction using FashnHumanParser with face/hair protection
    parser = await asyncio.to_thread(ClothingParser.get_instance)
    clothing_mask_arr, protect_mask_arr = await asyncio.to_thread(
        parser.extract_clothing_mask, raw_image, category=category
    )

    if np.sum(clothing_mask_arr > 0) < 300:
        raise HTTPException(status_code=422, detail="No suitable clothing region detected to replace.")

    ok_m, mask_png = cv2.imencode(".png", clothing_mask_arr)
    if not ok_m:
        raise HTTPException(status_code=500, detail="Could not encode clothing mask.")
    mask_bytes = mask_png.tobytes()

    nparr = np.frombuffer(raw_image, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    ok_p, person_png = cv2.imencode(".png", bgr_img)
    person_bytes = person_png.tobytes() if ok_p else raw_image

    pos_prompt = (
        f"RAW photo, ultra-realistic portrait of a person wearing {dress_prompt.strip()}, "
        "photorealistic fabric texture, natural folds, studio lighting, 8k uhd, dslr, masterpiece"
    )
    neg_prompt = (
        "nude, bare breasts, cleavage, bad anatomy, deformed limbs, blurry, low quality, "
        "artifacts, cartoon, distorted clothing seams, disfigured face, mutated"
    )

    async with GENERATION_LOCK:
        try:
            status = await check_comfy_status(force=True)
            if not status.get("ready"):
                detail = status.get("error") or "ComfyUI is not ready for generation."
                raise HTTPException(status_code=503, detail=detail)

            result = await run_clothing_inpaint(
                image_bytes=person_bytes,
                mask_bytes=mask_bytes,
                face_crop_png=profile.face_crop_png,
                positive_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                seed=secrets.randbelow(2 ** 32 - 1) + 1,
            )

            # Verification pass
            verification = await asyncio.to_thread(
                verify, profile.face_crop_png, result.image_bytes, profile.skin_lab
            )

            elapsed_ms = int((time.monotonic() - started) * 1000)
            return {
                "image_base64": "data:image/png;base64," + base64.b64encode(result.image_bytes).decode("ascii"),
                "applied_prompt": pos_prompt,
                "seed": int(result.seed),
                "width": int(result.width),
                "height": int(result.height),
                "checkpoint": result.checkpoint,
                "identity_similarity": float(verification["identity_similarity"]),
                "identity_match_percent": round(float(verification["identity_similarity"]) * 100.0, 1),
                "identity_verified": bool(verification["ok"]),
                "skin_delta_e": verification.get("skin_delta_e"),
                "skin_delta_chroma": verification.get("skin_delta_chroma"),
                "skin_match_percent": (
                    round(max(0.0, 100.0 - float(verification["skin_delta_chroma"]) * 5.0), 1)
                    if verification.get("skin_delta_chroma") is not None
                    else None
                ),
                "face_texture": verification.get("texture"),
                "elapsed_ms": elapsed_ms,
                "request_id": request_id,
                "mode": "dress-swap",
                "category": category,
            }
        except HTTPException:
            raise
        except FaceIdUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ComfyUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ComfyTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except OutOfMemory as exc:
            raise HTTPException(status_code=503, detail=f"The GPU ran out of memory: {exc}") from exc
        except ComfyGraphError as exc:
            raise HTTPException(status_code=502, detail=f"ComfyUI could not render this scene: {exc}") from exc
        except Exception as exc:
            request_log.error("dress swap failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail="Unexpected dress swap error; check server log.") from exc


@app.post("/api/generate")
async def api_generate(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    aspect: str = Form("4:5"),
    sliders: str = Form("{}"),
    wait: bool = Query(True),
) -> JSONResponse:
    request_id = _request_id(request)
    raw = await _read_upload(image)
    if aspect not in config.ASPECT_MAP:
        raise HTTPException(status_code=422, detail=f"Unsupported aspect {aspect}.")
    slider_dict = _parse_sliders(sliders)

    queue_mgr = QueueManager.get_instance()
    job = await queue_mgr.submit(
        "generate",
        _execute_generate,
        raw=raw,
        user_prompt=prompt,
        aspect=aspect,
        slider_dict=slider_dict,
        request_id=request_id,
    )

    if not wait:
        return JSONResponse(
            status_code=202,
            content={"job_id": job.job_id, "status": job.status.value, "position": job.position},
        )

    await job.event.wait()
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=job.error_status, detail=job.error or "Generation failed.")
    return JSONResponse(job.result)


@app.post("/api/dress-swap")
async def api_dress_swap(
    request: Request,
    image: UploadFile = File(...),
    dress_prompt: str = Form(...),
    category: str = Form("one-pieces"),
    garment_ref: Optional[UploadFile] = File(None),
    wait: bool = Query(True),
) -> JSONResponse:
    request_id = _request_id(request)
    raw = await _read_upload(image)
    ref_raw = None
    if garment_ref and garment_ref.filename:
        ref_raw = await _read_upload(garment_ref)

    if not dress_prompt or not dress_prompt.strip():
        raise HTTPException(status_code=422, detail="Please provide a description of the replacement dress.")

    if category not in ("one-pieces", "tops", "bottoms"):
        category = "one-pieces"

    queue_mgr = QueueManager.get_instance()
    job = await queue_mgr.submit(
        "dress-swap",
        _execute_dress_swap,
        raw_image=raw,
        dress_prompt=dress_prompt,
        category=category,
        garment_ref_bytes=ref_raw,
        request_id=request_id,
    )

    if not wait:
        return JSONResponse(
            status_code=202,
            content={"job_id": job.job_id, "status": job.status.value, "position": job.position},
        )

    await job.event.wait()
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=job.error_status, detail=job.error or "Dress swap generation failed.")
    return JSONResponse(job.result)


def _run_rescue(generated_png: bytes, profile: ReferenceProfile):
    try:
        from backend.face_swapper import FaceSwapper

        swapper = FaceSwapper.get_instance()
        rescued = swapper.swap_face(generated_png, profile.face_crop_png)
        rescued = grade_render(rescued, profile.skin_lab)
        metrics = verify(profile.face_crop_png, rescued, profile.skin_lab)
        logger.info(
            "rescue sim=%.3f texture=%.1f ok=%s",
            float(metrics.get("identity_similarity") or 0),
            float(metrics.get("texture") or 0),
            metrics.get("ok"),
        )
        return rescued, metrics
    except Exception as exc:
        logger.warning("identity rescue unavailable or failed: %s", exc)
        return None, None
