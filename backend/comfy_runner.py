import uuid
import random
import asyncio
import httpx
from backend.analyzer import prepare_face_reference

COMFY_URL = "http://127.0.0.1:8188"

ASPECT_MAP = {
    "4:5": (896, 1152),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
}


async def check_comfy_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{COMFY_URL}/object_info/CheckpointLoaderSimple")
            if r.status_code == 200:
                data = r.json()
                ckpts = (
                    data.get("CheckpointLoaderSimple", {})
                    .get("input", {})
                    .get("required", {})
                    .get("ckpt_name", [[]])[0]
                )
                return {"online": True, "checkpoints": ckpts}
    except Exception as e:
        return {"online": False, "error": str(e)}
    return {"online": False, "error": "ComfyUI not responding"}


async def run_comfy_generation(
    ref_image_bytes: bytes,
    positive_prompt: str,
    negative_prompt: str,
    aspect: str = "4:5",
) -> bytes:
    width, height = ASPECT_MAP.get(aspect, (896, 1152))

    # Prepare balanced 640x640 head-and-shoulders crop so selfie distance never warps proportions
    clean_face_bytes = prepare_face_reference(ref_image_bytes)

    async with httpx.AsyncClient(timeout=360.0) as client:
        obj_resp = await client.get(f"{COMFY_URL}/object_info")
        if obj_resp.status_code != 200:
            raise RuntimeError("ComfyUI server on port 8188 is not reachable.")
        obj_info = obj_resp.json()

        ckpts = (
            obj_info.get("CheckpointLoaderSimple", {})
            .get("input", {})
            .get("required", {})
            .get("ckpt_name", [[]])[0]
        )
        if not ckpts:
            raise RuntimeError("No SDXL checkpoint found in ComfyUI/models/checkpoints/.")
        ckpt_name = ckpts[0]

        has_faceid = (
            "IPAdapterUnifiedLoaderFaceID" in obj_info and "IPAdapterFaceID" in obj_info
        )

        files = {
            "image": ("reference_face_crop.png", clean_face_bytes, "image/png"),
            "overwrite": (None, "true"),
        }
        up_resp = await client.post(f"{COMFY_URL}/upload/image", files=files)
        up_resp.raise_for_status()
        uploaded_filename = up_resp.json()["name"]

        seed = random.randint(1, 2**48)
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt_name},
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": uploaded_filename},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
        }

        model_ref = ["1", 0]
        clip_ref = ["1", 1]

        if has_faceid:
            workflow["4"] = {
                "class_type": "IPAdapterUnifiedLoaderFaceID",
                "inputs": {
                    "model": model_ref,
                    "preset": "FACEID PLUS V2",
                    "lora_strength": 0.60,
                    "provider": "CPU",
                },
            }
            workflow["6"] = {
                "class_type": "IPAdapterFaceID",
                "inputs": {
                    "model": ["4", 0],
                    "ipadapter": ["4", 1],
                    "image": ["2", 0],
                    "weight": 0.85,
                    "weight_faceidv2": 0.55,
                    "weight_type": "ease in-out",
                    "combine_embeds": "concat",
                    "start_at": 0.10,
                    "end_at": 0.92,
                    "embeds_scaling": "K+V w/ C penalty",
                },
            }
            model_ref = ["6", 0]

        workflow["7"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": clip_ref},
        }
        workflow["8"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": clip_ref},
        }
        workflow["9"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 24,
                "cfg": 4.5,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": model_ref,
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["5", 0],
            },
        }
        workflow["10"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
        }
        workflow["11"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["10", 0], "filename_prefix": "Fotogenie_HQ"},
        }

        client_id = str(uuid.uuid4())
        prompt_resp = await client.post(
            f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": client_id}
        )
        if prompt_resp.status_code != 200:
            raise RuntimeError(f"ComfyUI Graph Error: {prompt_resp.text}")
        prompt_id = prompt_resp.json()["prompt_id"]

        for _ in range(300):
            await asyncio.sleep(1.0)
            hist_resp = await client.get(f"{COMFY_URL}/history/{prompt_id}")
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(
                            f"ComfyUI execution failed: {status.get('messages')}"
                        )
                    outputs = entry.get("outputs", {})
                    if "11" in outputs and outputs["11"].get("images"):
                        img_meta = outputs["11"]["images"][0]
                        view_resp = await client.get(
                            f"{COMFY_URL}/view",
                            params={
                                "filename": img_meta["filename"],
                                "subfolder": img_meta.get("subfolder", ""),
                                "type": img_meta.get("type", "output"),
                            },
                        )
                        view_resp.raise_for_status()
                        return view_resp.content

        raise RuntimeError("Timed out waiting for ComfyUI GPU generation.")
