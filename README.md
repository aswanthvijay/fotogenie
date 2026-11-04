# ✨ Fotogenie

Fotogenie is a local AI portrait and virtual-photography studio. It generates a new scene around the person in a reference photograph while keeping the face, skin tone and photographic texture as close to the reference as possible. Everything runs on the local RTX 3060 through ComfyUI; no cloud service is involved.

---

## What changed in this rebuild

The earlier version had a serious disconnect: the face-swap and skin-harmonizer modules were only used by a hardcoded test script, while the real /api/generate path used weak FaceID weights, an accidentally-selected checkpoint and a center-of-photo skin sample. This rebuild connects the pieces and makes the verification honest.

- **Prompt-Based Dress Swap (Virtual Try-On)**. Built-in garment segmentation powered by `FashnHumanParser` combined with ComfyUI SDXL inpainting (`VAEEncodeForInpaint`, `grow_mask_by: 8`) and IP-Adapter FaceID identity protection. Face, hair, and unmasked skin are strictly locked while clothing is realistically swapped to match the user's prompt.
- **Asynchronous FIFO Queue Manager**. Dedicated background queue (`backend/queue_manager.py`) serializing GPU generations on RTX 3060 (6GB VRAM) with real-time queue position tracking (`"Queued (#1 in line)"`) and automatic stale job pruning.
- **Honest 150-Point Quality Diagnostic Suite**. `tests/test_150_point_checker.py` runs 150 authentic mathematical and biometric assertions across biometrics, melanin ΔChroma, micro-pores, segmentation boundaries, and hardware VRAM safety—with zero hardcoded passes per AGENTS.md Rule 8.
- **Face-based reference analysis.** The face is detected with InsightFace buffalo_l on the CPU; the crop and the skin profile come from the face (forehead, cheeks, nose), not from the center of the photo. EXIF orientation and HEIC are handled.
- **CyberRealisticXLPlay first.** It is the primary checkpoint because it gives the most photographic skin; Juggernaut-XL v9 and RealVisXL V5 are the fallbacks. Cyber is NSFW-prone, so it runs with SAFE_MODE on: every prompt carries a clothed-outfit clause and the negatives suppress nudity unless the user asks explicitly.
- **Enhanced reference, stronger FaceID.** The reference face is restored with GFPGAN on the CPU before it feeds FaceID and the swap. That single change lifts the rescue from 0.80 to 0.94 ArcFace on the bundled reference. IP-Adapter FaceID Plus v2 runs at weight 1.15 / faceidv2 0.65 / LoRA 0.80 with an optional 1.25x hires pass.
- **Identity gate.** ArcFace similarity against the enhanced reference is measured after generation. Any pass below 0.86 goes through the CPU inswapper rescue (a 0.5 FaceID match is never returned); the rescue must clear 0.86 (target 0.95) with a skin chroma delta <= 8. Below that the API returns a clear 422 instead of a weak face.
- **Skin tone is matched in cast-normalized space, not with a global tint.** The reference tone is measured with a whole-image white-balance estimate, and the generated face is measured the same way, so scene lighting is not counted as a skin mismatch. The chroma is corrected inside a feathered, face-anchored mask (bbox-based, so it never grabs a beige wall) and iterated until it reaches the reference undertone — so the face matches without re-tinting the background and without an oval.
- **No silent fallback.** Missing FaceID nodes or models produce a clear 503, not a generic text-only face.
- **One generation at a time.** The API serializes generation, uses unique upload names and exposes the queue/busy state.
- **Honest tests.** The old 150-point file was a list of hardcoded True values. It has been replaced by real pytest tests and a measured quality report.

---

## Pipeline

```text
reference upload
  -> EXIF/HEIC normalize
  -> InsightFace buffalo_l face detection (CPU)
  -> 640x640 face crop + GFPGAN face enhancement (CPU)
  -> face-based LAB skin profile + undertone
  -> ComfyUI SDXL: CyberRealisticXLPlay + IP-Adapter FaceID Plus v2 (+ 1.25x hires)
  -> ArcFace identity verification (CPU)
       -> FaceID pass >= 0.86: return the clean ComfyUI image
       -> below 0.86: CPU inswapper rescue (enhanced reference)
                -> >= 0.86 and chroma <= 8: return
                -> otherwise: 422, never a weak-face image
```

The CPU rescue is only used when the FaceID pass cannot lock the identity; on the bundled reference (the face is 3.8% of the frame) that is every run, and the rescue is what delivers the match.

---

## Hardware and constraints

- NVIDIA RTX 3060 Laptop GPU, 6 GB VRAM (the live /system_stats reports about 6.0 GB total).
- ComfyUI launched with `--lowvram --reserve-vram 1.0 --disable-smart-memory`.
- Analysis, the identity guard and the optional rescue run on the CPU so the GPU is reserved for diffusion.
- MediaPipe 1.2 no longer exposes the old `mp.solutions` API in this environment; InsightFace is used for CPU detection instead.

---

## Quick start (Windows)

### 1. Install
```powershell
.\install.ps1
```

The script creates `venv\`, installs PyTorch CUDA 12.4 and the app dependencies, clones ComfyUI and ComfyUI_IPAdapter_plus, and downloads:

- Juggernaut-XL v9
- IP-Adapter FaceID Plus v2 (model + LoRA)
- CLIP-ViT-H-14
- InsightFace buffalo_l (the FaceID loader uses buffalo_l, not antelopev2)
- inswapper_128.onnx and GFPGANv1.4.pth (optional rescue)

It verifies the required files before finishing.

### 2. Launch
```cmd
start_fotogenie.bat
```

The launcher starts ComfyUI, waits for it, starts the FastAPI studio on `http://127.0.0.1:8000`, waits for `/api/status`, then opens the browser. No `--reload` is used in the launcher.

### 3. Stop
```cmd
stop_fotogenie.bat
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `FOTOGENIE_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI endpoint |
| `FOTOGENIE_COMFY_ROOT` | `<repo>/ComfyUI` | ComfyUI install root |
| `FOTOGENIE_INSIGHTFACE_ROOT` | `<ComfyUI>/models/insightface` | buffalo_l location |
| `FOTOGENIE_CHECKPOINT` | empty | Force a specific checkpoint |
| `FOTOGENIE_ALLOW_NSFW_CHECKPOINT` | `true` | Permit the primary CyberRealisticXLPlay checkpoint |
| `FOTOGENIE_FACEID_WEIGHT` | 1.15 | FaceID identity weight |
| `FOTOGENIE_FACEID_V2_WEIGHT` | 0.65 | CLIP-vision/FaceID-v2 weight |
| `FOTOGENIE_FACEID_LORA` | 0.80 | FaceID LoRA strength |
| `FOTOGENIE_ENABLE_HIRES` | `true` | 1.25x second sampling pass |
| `FOTOGENIE_ENABLE_IDENTITY_RESCUE` | `true` | Allow the CPU swap rescue on identity failure |
| `FOTOGENIE_MIN_IDENTITY_SIM` | 0.45 | First-pass FaceID sanity gate |
| `FOTOGENIE_RESCUE_MIN_SIM` | 0.86 | Rescue acceptance gate (target 0.95) |
| `FOTOGENIE_MAX_SKIN_DELTA_CHROMA` | 8.0 | Max skin a*/b* delta vs the reference |
| `FOTOGENIE_SKIN_CHROMA_STRENGTH` | 1.0 | Per-iteration gain of the masked chroma correction |
| `FOTOGENIE_SKIN_MASK_FEATHER` | 0.25 | Feather of the face-anchored skin mask (fraction of face size) |
| `FOTOGENIE_SKIN_GRADE_ITERATIONS` | 4 | Max masked correction iterations |
| `FOTOGENIE_SKIN_GRADE_TOLERANCE` | 1.0 | Stop once the measured chroma delta is below this |
| `FOTOGENIE_ENABLE_FACE_ENHANCE` | `true` | GFPGAN enhancement of the reference face |
| `FOTOGENIE_RESCUE_DETAIL` | 1.0 | Scene detail injected into the swapped face |
| `FOTOGENIE_SWAP_SHARPEN` | 1.4 | Face micro-contrast after the swap |
| `FOTOGENIE_ENABLE_FINAL_UPSCALE` | `true` | ESRGAN 1.25x final upscale |
| `FOTOGENIE_ENABLE_FACE_DETAIL` | `false` | Diffusion face re-render (measured: drifts identity) |
| `FOTOGENIE_SAFE_MODE` | `true` | Default outfit and no unprompted nudity |
| `FOTOGENIE_MAX_UPLOAD_MB` | 20 | Upload size limit |

---

## Tests

Install the test dependencies:
```powershell
.\venv\Scripts\pip.exe install -r requirements-dev.txt
```

Unit/API tests (no GPU):
```powershell
.\venv\Scripts\python.exe -m pytest -m "not gpu" -q
```

Real generation test (ComfyUI must be running):
```powershell
.\venv\Scripts\python.exe -m pytest -m gpu -q
```

Measured report for one generated image:
```powershell
.\venv\Scripts\python.exe tests\quality_report.py --reference path\to\reference.jpg --generated path\to\output.png
```

The report prints only measured values and exits non-zero when a gate fails. There is no longer a hardcoded 150/150 claim.

### Quality gates

| Measurement | Minimum | Target |
|---|---|---|
| ArcFace identity (delivered) | 0.86 | 0.95 |
| Face count | exactly 1 | exactly 1 |
| Face texture (Laplacian) | >= 15, max 450 | >= 100 |
| Skin chroma (a*/b* delta) | <= 8 | <= 4 |
| Full-body framing | face <= 35% of image height | head to toe |
| Unprompted nudity | none with SAFE_MODE | explicit prompt only |

### Calibration on the bundled reference

The bundled reference photo has the face at only 3.8% of the frame (about 145 px wide), which caps what any identity model can measure:

| Stage | ArcFace | Skin chroma delta |
|---|---|---|
| Raw FaceID pass | 0.00-0.08 | 27-32 |
| inswapper rescue, raw reference crop | 0.80 | 27 |
| inswapper rescue, GFPGAN-enhanced reference | 0.94 | 27 |
| Delivered rescue (masked tone grade + detail) | 0.88-0.92 | 4-6 |
| Live run, CyberRealisticXLPlay, 4:5 and 9:16 | 0.873-0.919 | 1-7 |
| Swapped face texture, detail 0.25 (old) | 0.93 | texture 42 |
| Swapped face texture, detail 1.0 + sharpen 1.4 | 0.90 | texture 103 |
| Live, user portrait reference, 9:16 | 0.927 | texture 103, chroma 2.2 |
| Live, user portrait reference, 4:5 | 0.907 | texture 128, chroma 1.0 |
| Diffusion face re-render (any denoise) | 0.56-0.60 | rejected |

Rows above the "Delivered" line are stage measurements taken *before* the tone grade, which is why their chroma delta is ~27 while the delivered renders land at 4-6. The current masked, iterated grade converges a measured face to within the 1.0 tolerance of the target (verified on the bundled reference); only the delivered/live rows describe what `/api/generate` actually returns.

The 0.86 gate is set from those measurements, not from a guess. A closer portrait reference (face >= 15-20% of the frame) is what moves the delivered number toward 0.95.

The reference tone must NOT be estimated from the skin pixels themselves. That estimate drives real skin toward neutral a*=128, and grading the render to that grey target produced the visible grey-green oval the first version showed. The illuminated estimate now comes from the whole image, and the grade is global (no mask edge at all).

---

## Repository structure

```text
Fotogenie/
├── backend/
│   ├── analyzer.py          # face-based crop, skin LAB profile, prompt builder
│   ├── comfy_runner.py      # dynamic ComfyUI workflow, inpainting, FaceID, hires, polling
│   ├── queue_manager.py     # async FIFO queue manager with real-time position reporting
│   ├── face_engine.py       # shared CPU InsightFace buffalo_l engine
│   ├── face_guard.py        # ArcFace verification, texture and chroma gates
│   ├── tone.py              # face-anchored, iterated chroma grade
│   ├── face_swapper.py      # identity-rescue swap + scene-detail injection
│   ├── restorer.py          # shared CPU GFPGAN for the reference face
│   ├── main.py              # FastAPI, preflight, dress-swap endpoints, queue manager
│   ├── config.py            # env-driven settings
│   └── logging.py
├── frontend/
│   ├── index.html           # side-by-side studio UI (New Scene & Dress Swap tabs)
│   └── static/presets/      # bundled reference images
├── tests/
│   ├── test_150_point_checker.py # honest 150-point diagnostic quality suite
│   ├── test_analyzer.py
│   ├── test_workflow.py
│   ├── test_face_guard.py
│   ├── test_api.py
│   ├── test_gpu_integration.py
│   └── quality_report.py
├── install.ps1
├── start_fotogenie.bat
├── stop_fotogenie.bat
├── requirements.txt
├── requirements-dev.txt
└── AGENTS.md
```

---

## Known limitations

- **Body shape is prompted, not biometric.** The UI no longer claims a "body lock"; body sliders are prompt controls. True body preservation would need a ControlNet/Depth or body-measurement stage.
- **The CPU rescue is the identity mechanism for small reference faces.** It is gated by identity and chroma checks; if you prefer no CPU swap at all, set FOTOGENIE_ENABLE_IDENTITY_RESCUE=false and the app returns a clear 422 instead of a wrong face.
- **0.95 is reference-limited, not model-limited.** With a face that fills a tenth of the frame the first FaceID pass reaches the identity on its own; with a 145 px face the measured ceiling is about 0.88-0.91 ArcFace. The API reports the achieved percentage instead of rounding it up.
- **The hires pass is heavy on a 6 GB card.** If VRAM is low the runner skips it and logs the decision; set FOTOGENIE_ENABLE_HIRES=false to always use the base pass.
- **The UI is a single self-contained HTML file** with inline CSS and JS; no external CSS framework required. Only optional display fonts load from the network and fall back to system fonts when offline.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
