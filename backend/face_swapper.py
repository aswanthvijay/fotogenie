import os
import cv2
import numpy as np
import torch
import spandrel
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper
from insightface.utils import face_align

INSWAPPER_PATH = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
GFPGAN_PATH = os.path.expanduser("~/.insightface/models/GFPGANv1.4.pth")

def match_skin_colors(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Matches the color of source skin to target skin in LAB color space avoiding hair and background."""
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 512x512 face skin ellipse (covers cheeks, nose, forehead; excludes hair and dark background)
    h, w = source_bgr.shape[:2]
    skin_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(skin_mask, (int(w*0.5), int(h*0.53)), (int(w*0.26), int(h*0.32)), 0, 0, 360, 255, -1)
    
    # Exclude dark hair/eyes within ellipse (L > 60)
    src_skin = (skin_mask > 0) & (src_lab[..., 0] > 60)
    tgt_skin = (skin_mask > 0) & (tgt_lab[..., 0] > 60)
    
    if np.count_nonzero(src_skin) < 100 or np.count_nonzero(tgt_skin) < 100:
        return source_bgr

    out_lab = src_lab.copy()
    for i in range(3):
        src_mean = float(np.mean(src_lab[..., i][src_skin]))
        src_std = float(np.std(src_lab[..., i][src_skin])) + 1e-6
        tgt_mean = float(np.mean(tgt_lab[..., i][tgt_skin]))
        tgt_std = float(np.std(tgt_lab[..., i][tgt_skin])) + 1e-6
        
        # Reinhard transfer on skin distribution
        out_lab[..., i] = np.clip((src_lab[..., i] - src_mean) * (tgt_std / src_std) + tgt_mean, 0, 255)
    
    # Feathered blend with original so eyes, teeth, and hair retain their natural contrast and specular highlights
    feathered_mask = cv2.GaussianBlur(skin_mask, (41, 41), 0).astype(np.float32) / 255.0
    feathered_mask = feathered_mask[..., None]
    
    final_lab = out_lab * feathered_mask + src_lab * (1.0 - feathered_mask)
    return cv2.cvtColor(final_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def clean_unnatural_makeup(face_roi: np.ndarray) -> np.ndarray:
    """Removes synthetic blue/cyan/turquoise eyeliner and eyeshadow artifacts from facial dermis."""
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    # Detect unnatural blue/cyan/turquoise tones in facial skin (H: 85 to 145)
    blue_mask = cv2.inRange(hsv, np.array([85, 35, 25]), np.array([145, 255, 255]))
    if np.count_nonzero(blue_mask) < 20:
        return face_roi
    
    feathered = cv2.GaussianBlur(blue_mask, (9, 9), 0).astype(np.float32) / 255.0
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Neutralize to natural dark lash / dermal color
    lab[..., 1] = lab[..., 1] * (1.0 - feathered) + 137.0 * feathered
    lab[..., 2] = lab[..., 2] * (1.0 - feathered) + 128.0 * feathered
    # Deepen to natural dark kohl / mascara tone
    lab[..., 0] = np.clip(lab[..., 0] * (1.0 - feathered * 0.40), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

class FaceSwapper:
    _instance = None

    def __init__(self):
        # Initialize FaceAnalysis on CPU so 100% GPU VRAM is reserved for ComfyUI
        self.app = FaceAnalysis(
            name='buffalo_l',
            root='D:/Fotogenie/ComfyUI/models/insightface',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load biometric swap model
        if os.path.exists(INSWAPPER_PATH):
            self.swapper = INSwapper(INSWAPPER_PATH)
        else:
            self.swapper = None

        # Load ultra-high-clarity face restoration model (GFPGAN 512x512)
        if os.path.exists(GFPGAN_PATH):
            try:
                descriptor = spandrel.ModelLoader().load_from_file(GFPGAN_PATH)
                self.gfpgan = descriptor.model.to("cpu").eval()
            except Exception as e:
                print(f"[FaceSwapper] Could not load GFPGAN: {e}")
                self.gfpgan = None
        else:
            self.gfpgan = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def swap_face(self, gen_image_bytes: bytes, ref_image_bytes: bytes, target_lab: list = None) -> bytes:
        if self.swapper is None:
            return gen_image_bytes
        try:
            nparr_gen = np.frombuffer(gen_image_bytes, np.uint8)
            gen_bgr = cv2.imdecode(nparr_gen, cv2.IMREAD_COLOR)

            nparr_ref = np.frombuffer(ref_image_bytes, np.uint8)
            ref_bgr = cv2.imdecode(nparr_ref, cv2.IMREAD_COLOR)

            if gen_bgr is None or ref_bgr is None:
                return gen_image_bytes

            ref_faces = self.app.get(ref_bgr)
            gen_faces = self.app.get(gen_bgr)

            if not ref_faces or not gen_faces:
                return gen_image_bytes

            ref_face = sorted(ref_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[0]
            gen_face = sorted(gen_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[0]

            # Capture both the original generated face and the reference face crop for 100% skin matching
            orig_aimg512, _ = face_align.norm_crop2(gen_bgr, gen_face.kps, 512)
            ref_aimg512, _ = face_align.norm_crop2(ref_bgr, ref_face.kps, 512)

            # 1. High-precision biometric face swap with native difference-gradient blending
            final_bgr = self.swapper.get(gen_bgr, gen_face, ref_face, paste_back=True)
            if final_bgr is None:
                return gen_image_bytes

            # 2. 512x512 Face Restoration to match SDXL body resolution (pores, crisp eyes, teeth)
            if self.gfpgan is not None:
                try:
                    swapped_faces = self.app.get(final_bgr)
                    if swapped_faces:
                        target_f = sorted(swapped_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[0]
                        aimg512, M512 = face_align.norm_crop2(final_bgr, target_f.kps, 512)
                    else:
                        aimg512, M512 = face_align.norm_crop2(final_bgr, gen_face.kps, 512)
                        
                    rgb = cv2.cvtColor(aimg512, cv2.COLOR_BGR2RGB)
                    t_in = (torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5) - 1.0
                    with torch.no_grad():
                        out = self.gfpgan(t_in)
                        restored_t = out[0] if isinstance(out, (list, tuple)) else out
                    restored = np.clip((restored_t.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

                    # 100% Skin Tone Match: Align restored face color directly to the user's reference face dermis
                    restored_bgr = match_skin_colors(restored_bgr, ref_aimg512)

                    # Balanced 65% restoration blend to preserve genuine cellular skin pores and eliminate any plastic/doll look
                    blended_512 = cv2.addWeighted(restored_bgr, 0.65, aimg512, 0.35, 0)

                    # High-pass micro-pore injection to guarantee authentic living human skin texture
                    ref_gray = cv2.cvtColor(ref_aimg512, cv2.COLOR_BGR2GRAY)
                    ref_blur = cv2.GaussianBlur(ref_gray, (0, 0), 2.2)
                    high_pass_pores = cv2.subtract(ref_gray, ref_blur)
                    pores_3d = np.dstack([high_pass_pores, high_pass_pores, high_pass_pores]).astype(np.float32)
                    blended_512 = np.clip(blended_512.astype(np.float32) + (pores_3d - 128.0) * 0.18, 0, 255).astype(np.uint8)

                    # Smooth feathered elliptical mask to eliminate any seam at hairline or jaw
                    mask = np.zeros((512, 512), dtype=np.float32)
                    cv2.ellipse(mask, (256, 270), (170, 200), 0, 0, 360, 1.0, -1)
                    mask = cv2.GaussianBlur(mask, (51, 51), 17)[:, :, None]

                    IM = cv2.invertAffineTransform(M512)
                    h, w = final_bgr.shape[:2]
                    warped_face = cv2.warpAffine(blended_512, IM, (w, h), borderValue=0.0)
                    warped_mask = cv2.warpAffine(mask, IM, (w, h), borderValue=0.0)[:, :, None]
                    final_bgr = (warped_mask * warped_face.astype(np.float32) + (1.0 - warped_mask) * final_bgr.astype(np.float32)).astype(np.uint8)
                except Exception as ex:
                    print(f"[FaceSwapper] GFPGAN restoration skipped: {ex}")

            # 3. Full-Body 100% Skin Tone Harmonization (aligns face, neck, arms, and hands to reference melanin)
            try:
                from backend.skin_harmonizer import SkinHarmonizer
                harmonizer = SkinHarmonizer.get_instance()
                final_bgr = harmonizer.harmonize_skin_100pct(final_bgr, ref_bgr, strength=1.0)
                
                # Apply subtle organic 35mm photographic sensor grain to eliminate any plastic AI look
                skin_mask = harmonizer.get_skin_mask(final_bgr)
                final_bgr = harmonizer.apply_organic_film_grain(final_bgr, skin_mask, grain_scale=1.5)
            except Exception as e_harm:
                print(f"[FaceSwapper] Skin harmonization skipped: {e_harm}")

            # 3. Localized micro-texture boost for razor-sharp iris catchlights and eyelashes
            try:
                b = [int(x) for x in gen_face.bbox]
                pad = 15
                h, w, _ = final_bgr.shape
                y1, y2 = max(0, b[1]-pad), min(h, b[3]+pad)
                x1, x2 = max(0, b[0]-pad), min(w, b[2]+pad)
                if y2 > y1 and x2 > x1:
                    roi = clean_unnatural_makeup(final_bgr[y1:y2, x1:x2])
                    blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
                    sharpened = cv2.addWeighted(roi, 1.35, blur, -0.35, 0)
                    
                    # Soft boundary feathering to guarantee zero rectangular edge artifacts
                    mask_roi = np.ones(roi.shape[:2], dtype=np.float32)
                    rh, rw = roi.shape[:2]
                    pad_feather = min(15, rh // 4, rw // 4)
                    if pad_feather > 2:
                        mask_roi[:pad_feather, :] = 0
                        mask_roi[-pad_feather:, :] = 0
                        mask_roi[:, :pad_feather] = 0
                        mask_roi[:, -pad_feather:] = 0
                        mask_roi = cv2.GaussianBlur(mask_roi, (21, 21), 7)[:, :, None]
                        final_bgr[y1:y2, x1:x2] = (mask_roi * sharpened.astype(np.float32) + (1.0 - mask_roi) * roi.astype(np.float32)).astype(np.uint8)
                    else:
                        final_bgr[y1:y2, x1:x2] = sharpened
            except Exception:
                pass

            _, encoded = cv2.imencode(".png", final_bgr)
            return encoded.tobytes()
        except Exception as e:
            print(f"[FaceSwapper] Warning: Swap skipped ({e})")
            return gen_image_bytes
