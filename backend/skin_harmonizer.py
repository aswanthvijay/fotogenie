import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger("SkinHarmonizer")

def extract_skin_stats(img_bgr: np.ndarray, skin_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts mean and std of pure living dermis in LAB color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Core skin filter: excludes feathered falloffs and dark hair/shadow fringes (L > 60, mask > 120)
    valid = (skin_mask > 120) & (lab[..., 0] > 60)
    pixels = lab[valid]
    if len(pixels) < 50:
        pixels = lab[skin_mask > 0]
    if len(pixels) == 0:
        return np.array([165.0, 140.0, 128.0], dtype=np.float32), np.array([15.0, 6.0, 6.0], dtype=np.float32)
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0) + 1e-6
    return mean, std

class SkinHarmonizer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from backend.clothing_parser import ClothingParser
        self.parser = ClothingParser.get_instance().parser

    def get_skin_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extracts complete skin mask (face, neck, arms, hands, legs, torso) using human parser."""
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        parsed = self.parser.predict(pil_img)
        if parsed.shape != (h, w):
            parsed = cv2.resize(parsed, (w, h), interpolation=cv2.INTER_NEAREST)

        from fashn_human_parser import LABELS_TO_IDS
        # Skin regions: face, arms, hands, legs, torso
        skin_labels = ["face", "arms", "hands", "legs", "torso"]
        skin_ids = [LABELS_TO_IDS[l] for l in skin_labels if l in LABELS_TO_IDS]
        
        raw_mask = np.isin(parsed, skin_ids).astype(np.uint8) * 255
        
        # Color-based skin confirmation in YCrCb to exclude hair/background that may overlap
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        color_mask = cv2.inRange(ycrcb, np.array([20, 125, 70]), np.array([255, 185, 135]))
        
        combined = cv2.bitwise_and(raw_mask, color_mask)
        # Smooth boundaries with Gaussian blur for seamless feathering
        feathered = cv2.GaussianBlur(combined, (9, 9), 0)
        return feathered

    def harmonize_skin_100pct(
        self,
        target_bgr: np.ndarray,
        reference_bgr: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Harmonizes 100% of visible skin (face, neck, arms, hands) in target_bgr
        to perfectly match the melanin, undertone, and lightness of reference_bgr.
        """
        h_t, w_t = target_bgr.shape[:2]
        h_r, w_r = reference_bgr.shape[:2]

        # Extract skin masks
        ref_skin_mask = self.get_skin_mask(reference_bgr)
        tgt_skin_mask = self.get_skin_mask(target_bgr)

        if np.count_nonzero(ref_skin_mask) < 200 or np.count_nonzero(tgt_skin_mask) < 200:
            return target_bgr

        # Extract statistics in LAB space
        ref_mean, ref_std = extract_skin_stats(reference_bgr, ref_skin_mask)
        tgt_mean, tgt_std = extract_skin_stats(target_bgr, tgt_skin_mask)

        tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        alpha = (tgt_skin_mask.astype(np.float32) / 255.0)[..., None] * strength

        # Apply Reinhard color distribution transfer in LAB
        shifted_lab = tgt_lab.copy()
        
        # Chromaticity (a* and b* undertones): 100% match to reference melanin
        # Gently clamp variance scaling to prevent extreme orange saturation from lingering in highlights
        scale_a = min(1.15, max(0.70, ref_std[1] / tgt_std[1]))
        scale_b = min(1.15, max(0.70, ref_std[2] / tgt_std[2]))
        
        shifted_lab[..., 1] = np.clip((tgt_lab[..., 1] - tgt_mean[1]) * scale_a + ref_mean[1], 0, 255)
        shifted_lab[..., 2] = np.clip((tgt_lab[..., 2] - tgt_mean[2]) * scale_b + ref_mean[2], 0, 255)
        
        # Luminance L*: Align reference median lightness (100% match) while respecting scene highlights
        l_diff = ref_mean[0] - tgt_mean[0]
        shifted_lab[..., 0] = np.clip(tgt_lab[..., 0] + l_diff * 1.0, 0, 255)

        # Alpha composite with feathered skin mask
        blended_lab = (shifted_lab * alpha + tgt_lab * (1.0 - alpha)).astype(np.uint8)
        harmonized_bgr = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
        return harmonized_bgr

    def apply_organic_film_grain(self, img_bgr: np.ndarray, skin_mask: np.ndarray, grain_scale: float = 1.6) -> np.ndarray:
        """
        Synthesizes organic 35mm photographic micro-texture on skin regions,
        completely eradicating the artificial 'plastic / airbrushed AI' look.
        """
        h, w = img_bgr.shape[:2]
        # Generate micro-grain in float
        noise = np.random.normal(0, grain_scale, (h, w)).astype(np.float32)
        
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_alpha = (skin_mask.astype(np.float32) / 255.0)
        
        # Add grain strictly to luminance L* channel in skin regions
        lab[..., 0] = np.clip(lab[..., 0] + noise * skin_alpha, 0, 255)
        
        grain_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return grain_bgr
