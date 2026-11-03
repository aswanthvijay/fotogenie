import logging
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List
from fashn_human_parser import FashnHumanParser, LABELS_TO_IDS

logger = logging.getLogger("ClothingParser")

class ClothingParser:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing FashnHumanParser singleton...")
        self.parser = FashnHumanParser()
        logger.info("FashnHumanParser initialized.")

    def extract_clothing_mask(
        self,
        image_bytes: bytes,
        category: str = "one-pieces",
        dilate_px: int = 14,
        feather_px: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract binary clothing mask and anti-mask from image bytes.
        Returns:
            clothing_mask (uint8, 0 or 255): Mask where clothing should be replaced.
            protect_mask (uint8, 0 or 255): Mask for face and hair which must never be touched.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        w_orig, h_orig = pil_img.size

        parsed = self.parser.predict(pil_img)
        if parsed.shape != (h_orig, w_orig):
            parsed = cv2.resize(parsed, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Map categories to labels
        if category == "tops":
            target_labels = ["top", "torso", "scarf"]
        elif category == "bottoms":
            target_labels = ["pants", "skirt", "belt"]
        else: # "one-pieces" or full dress
            target_labels = ["top", "dress", "skirt", "pants", "torso", "belt", "scarf"]

        clothing_ids = [LABELS_TO_IDS[l] for l in target_labels if l in LABELS_TO_IDS]
        raw_clothing_mask = np.isin(parsed, clothing_ids).astype(np.uint8) * 255

        # Face & hair protection
        protect_ids = [LABELS_TO_IDS[l] for l in ["face", "hair"] if l in LABELS_TO_IDS]
        protect_mask = np.isin(parsed, protect_ids).astype(np.uint8) * 255
        protect_dilated = cv2.dilate(protect_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

        # Dilate clothing mask to cover original seams, collars, and borders
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
            clothing_mask = cv2.dilate(raw_clothing_mask, k, iterations=1)
        else:
            clothing_mask = raw_clothing_mask

        # Strictly protect face & hair
        clothing_mask[protect_dilated > 0] = 0

        return clothing_mask, protect_mask

