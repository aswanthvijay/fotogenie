import base64
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.analyzer import extract_reference_profile, build_conditioned_prompt
from backend.comfy_runner import check_comfy_status, run_comfy_generation

app = FastAPI(title="Fotogenie Simple Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_HTML = Path("D:/Fotogenie/frontend/index.html")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FRONTEND_HTML.read_text(encoding="utf-8")


@app.get("/api/status")
async def api_status():
    return await check_comfy_status()


@app.post("/api/generate")
async def api_generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    aspect: str = Form("4:5"),
    sliders: str = Form("{}"),
):
    try:
        ref_bytes = await image.read()
        slider_dict = json.loads(sliders)
        profile = extract_reference_profile(ref_bytes)
        cond = build_conditioned_prompt(prompt, profile, slider_dict)

        final_bytes = await run_comfy_generation(
            ref_image_bytes=ref_bytes,
            positive_prompt=cond["positive_prompt"],
            negative_prompt=cond["negative_prompt"],
            aspect=aspect,
        )

        b64_img = base64.b64encode(final_bytes).decode("utf-8")
        return JSONResponse(
            {
                "image_base64": f"data:image/png;base64,{b64_img}",
                "applied_prompt": cond["positive_prompt"],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
