# AGENTS.md — Fotogenie Clean Rebuild Rules
1. ZERO FAKE FALLBACK LAW: Never return, copy, or overlay text on the user's input image. Every generation MUST run through ComfyUI (`http://127.0.0.1:8188`) on the RTX 3060 GPU.
2. DYNAMIC VALIDATION: Always query `GET http://127.0.0.1:8188/object_info` before submitting a workflow.
3. CONSUMER SIMPLICITY: Two clean side-by-side image cards (`object-contain`), zero technical jargon, optional body sliders hidden in a drawer by default.
4. HARDWARE: Single venv at `D:\Fotogenie\venv`, CPU-only MediaPipe/OpenCV analysis, ComfyUI launched with `--lowvram --reserve-vram 1.0 --disable-smart-memory`.
