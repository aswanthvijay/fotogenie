# AGENTS.md — Fotogenie Clean Rebuild Rules
1. ZERO FAKE FALLBACK LAW: Never return, copy, or overlay text on the user's input image. Every generation MUST run through ComfyUI (`http://127.0.0.1:8188`) on the RTX 3060 GPU.
2. DYNAMIC VALIDATION: Always query `GET http://127.0.0.1:8188/object_info` before submitting a workflow.
3. CONSUMER SIMPLICITY: Two clean side-by-side image cards (`object-contain`), zero technical jargon, optional body sliders hidden in a drawer by default.
4. HARDWARE: Single venv at `D:\Fotogenie\venv`, CPU-only InsightFace/OpenCV analysis (MediaPipe 1.2 no longer exposes the solutions API), ComfyUI launched with `--lowvram --reserve-vram 1.0 --disable-smart-memory`.
5. FACE-BASED ANALYSIS: Reference skin tone and cropping MUST come from the detected face, never from the center of the photograph. Reject uploads with no usable face.
6. EXPLICIT CHECKPOINT: Never use `ckpts[0]`. CyberRealisticXLPlay is the primary (user requirement for photographic skin), then Juggernaut-XL v9, then RealVisXL V5. Cyber is NSFW-prone, so `FOTOGENIE_ALLOW_NSFW_CHECKPOINT=true` is the default and SAFE_MODE must stay on: every prompt carries a clothed-outfit clause and nudity negatives unless the user asks explicitly.
7. IDENTITY IS REQUIRED: IP-Adapter FaceID and its models must be present; never silently fall back to text-only. After generation, verify with ArcFace. Never return an unverified wrong-face image.
8. HONEST TESTS: No hardcoded passes. A check either computes a real metric or is marked SKIP. The 150-point report is not a 100% claim unless every point was measured.
9. SAFE DEFAULT: Default to a clothed outfit and no nudity for benign prompts. An explicit nude/lingerie prompt is the only override.
