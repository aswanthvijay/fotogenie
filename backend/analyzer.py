import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import cv2
import numpy as np
from PIL import Image


def prepare_face_reference(image_bytes: bytes) -> bytes:
    """
    Crops a balanced head-and-shoulders square around the detected face so
    InsightFace and IP-Adapter preserve true jawline width, hair volume,
    and head-to-shoulder proportions without pasting cropped selfie edges.
    """
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        cx = fx + fw // 2
        cy = fy + fh // 2
        box_size = int(max(fw, fh) * 2.4)

        x1 = cx - box_size // 2
        y1 = cy - int(box_size * 0.52)
        x2 = x1 + box_size
        y2 = y1 + box_size

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            img_np = cv2.copyMakeBorder(
                img_np,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
            x1 += pad_left
            x2 += pad_left
            y1 += pad_top
            y2 += pad_top

        cropped_np = img_np[y1:y2, x1:x2]
        cropped = Image.fromarray(cropped_np)
    else:
        side = min(w, h)
        left = (w - side) // 2
        top = max(0, int((h - side) * 0.15))
        cropped = img_pil.crop((left, top, left + side, top + side))

    cropped = cropped.resize((640, 640), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def extract_reference_profile(image_bytes: bytes) -> dict:
    """
    Samples central facial skin on CPU to generate a natural skin-tone prompt
    anchor without applying destructive post-render pixel masks.
    """
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(img_pil)
    h, w, _ = img_rgb.shape

    center_zone = img_rgb[int(h * 0.25) : int(h * 0.55), int(w * 0.35) : int(w * 0.65)]
    ycrcb = cv2.cvtColor(center_zone, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([45, 135, 80]), np.array([235, 172, 125]))

    lab = cv2.cvtColor(center_zone, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_pixels = lab[mask > 0]
    if len(skin_pixels) < 50:
        skin_pixels = lab.reshape(-1, 3)

    med_l, med_a, med_b = np.median(skin_pixels, axis=0)

    if med_l > 195:
        skin_desc = "fair warm ivory complexion, even natural skin texture and pores"
    elif med_l > 168:
        skin_desc = "warm golden olive skin tone, even natural complexion with fine pores"
    elif med_l > 132:
        skin_desc = (
            "warm medium-brown South Asian Indian skin tone, smooth even caramel "
            "complexion, authentic natural skin micro-texture"
        )
    elif med_l > 100:
        skin_desc = (
            "deep warm rich brown South Asian melanin complexion, even natural skin tone"
        )
    else:
        skin_desc = "deep rich dark melanin complexion, even natural skin texture"

    return {
        "target_lab": [float(med_l), float(med_a), float(med_b)],
        "skin_descriptor": skin_desc,
        "body_descriptor": (
            "natural head-to-shoulder proportions, authentic facial jawline and cheek "
            "fullness, balanced anatomical figure"
        ),
    }


def build_conditioned_prompt(user_prompt: str, profile: dict, sliders: dict) -> dict:
    p_low = user_prompt.lower()
    tokens = [user_prompt.strip()]

    # 1. Skin Tone (Locked to reference unless overridden in prompt)
    skin_override = any(
        k in p_low
        for k in [
            "pale skin",
            "fair skin",
            "white skin",
            "dark skin",
            "ebony skin",
            "tanned skin",
            "bronze skin",
            "olive skin",
        ]
    )
    if not skin_override:
        tokens.append(f"({profile['skin_descriptor']}:1.15)")

    # 2. Height Override
    h_val = float(sliders.get("height", 0))
    if any(k in p_low for k in ["taller", "tall body", "long legs", "6ft"]) or h_val > 0.3:
        tokens.append("(tall stature, elongated body proportions:1.2)")
    elif any(k in p_low for k in ["short", "petite", "compact height"]) or h_val < -0.3:
        tokens.append("(petite compact height, shorter stature:1.2)")

    # 3. Bust Volume Override
    b_val = float(sliders.get("bust", 0))
    if (
        any(
            k in p_low
            for k in [
                "larger bust",
                "big bust",
                "fuller bust",
                "large breasts",
                "huge bust",
                "busty",
            ]
        )
        or b_val > 0.3
    ):
        w_boost = min(1.4, 1.15 + abs(b_val) * 0.12)
        tokens.append(f"(fuller bust, curvy chest proportions:{w_boost:.2f})")
    elif (
        any(k in p_low for k in ["small bust", "flat chest", "petite chest", "smaller bust"])
        or b_val < -0.3
    ):
        tokens.append("(slender petite bust:1.2)")

    # 4. Waist Width Override
    w_val = float(sliders.get("waist", 0))
    if (
        any(k in p_low for k in ["narrow waist", "tiny waist", "slim waist", "hourglass waist"])
        or w_val < -0.3
    ):
        tokens.append("(narrow hourglass waist, defined midsection:1.2)")
    elif any(k in p_low for k in ["wider waist", "thick waist", "broad waist"]) or w_val > 0.3:
        tokens.append("(natural fuller waist:1.2)")

    # 5. Hips / Glutes Override
    hip_val = float(sliders.get("hips", 0))
    if (
        any(
            k in p_low
            for k in ["wide hips", "curvy hips", "larger hips", "thick thighs", "curvy figure"]
        )
        or hip_val > 0.3
    ):
        w_boost = min(1.4, 1.15 + abs(hip_val) * 0.12)
        tokens.append(f"(wide curvy hips, full lower body proportions:{w_boost:.2f})")
    elif any(k in p_low for k in ["narrow hips", "slim hips", "straight hips"]) or hip_val < -0.3:
        tokens.append("(slender straight hips:1.2)")

    # 6. Overall Build Override
    build_val = float(sliders.get("build", 0))
    if (
        any(
            k in p_low
            for k in ["chubby", "plump", "plus size", "curvy build", "full figured", "thick body"]
        )
        or build_val > 0.3
    ):
        tokens.append("(full-figured soft natural body build:1.2)")
    elif (
        any(k in p_low for k in ["skinny", "slender", "thin build", "lean athletic"])
        or build_val < -0.3
    ):
        tokens.append("(slender lean body build:1.2)")
    else:
        tokens.append(profile["body_descriptor"])

    tokens.append(
        "unified natural scene lighting across face neck and shoulders, sharp symmetrical eyes, "
        "clean natural teeth, detailed realistic ear and jewelry, realistic fabric weave and folds, "
        "shot on Canon EOS R5 85mm f/1.8 lens, raw unretouched photograph, natural skin pores"
    )
    positive = ", ".join([t for t in tokens if t])

    negative = (
        "pasted face, face mask halo, dark jawline ring, white edge halo, glowing shoulder outline, "
        "oversized head, narrow pointed chin, blotchy skin, orange skin patches, grey lips, "
        "desaturated mouth, asymmetrical eyes, cross-eyed, fused teeth, blurry teeth, deformed ear, "
        "melted earring, plastic skin, waxy skin, airbrushed, cartoon, 3d render, CGI, bad anatomy, "
        "deformed hands, blurry, watermark"
    )

    return {
        "positive_prompt": positive,
        "negative_prompt": negative,
    }
