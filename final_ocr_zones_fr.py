import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
from paddleocr import PaddleOCR


# -------------------------------------------------------------
# PaddleOCR 2.6 initialization (DETECTION ONLY)
# -------------------------------------------------------------
text_detector = PaddleOCR(
    lang="ch",        # Chinese model for best detection
    use_gpu=False,
    show_log=False
)


# -------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------
def auto_rotate(img):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def load_image(image_path):
    img = cv2.imread(image_path)
    img = auto_rotate(img)
    return img


# -------------------------------------------------------------
# Draw detected boxes
# -------------------------------------------------------------
def draw_detected_boxes(img, ocr_result, save_path="/app/detected_boxes.jpg"):
    img = img.copy()
    boxes = ocr_result[0] if ocr_result else []

    for idx, box in enumerate(boxes):
        pts = []
        for pt in box:
            if isinstance(pt, list) and len(pt) == 2:
                pts.append([int(pt[0]), int(pt[1])])

        if len(pts) != 4:
            continue

        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_np], isClosed=True, color=(0,255,0), thickness=3)

        x, y = pts[0]
        cv2.putText(img, str(idx+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

    cv2.imwrite(save_path, img)
    print(f"[+] Detection visualization saved → {save_path}")


# -------------------------------------------------------------
# DETECT TEXT ZONES (No preprocessing)
# -------------------------------------------------------------
def detect_text_zones(img):
    detected_zones = []

    # Direct detection without preprocessing
    temp_path = "/app/_temp_input.jpg"
    cv2.imwrite(temp_path, img)
    
    ocr_result = text_detector.ocr(temp_path, det=True, rec=False, cls=False)

    print("\n=== RAW DETECTION ===")
    print(ocr_result)

    draw_detected_boxes(img, ocr_result)

    idx = 1
    boxes = ocr_result[0] if ocr_result else []

    for box in boxes:
        pts = []
        for pt in box:
            if isinstance(pt, list) and len(pt) == 2:
                pts.append([int(pt[0]), int(pt[1])])

        if len(pts) != 4:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

        h, w = crop.shape[:2]
        if h < 10 or w < 40:
            continue

        out_path = f"/app/zone_{idx}.jpg"
        cv2.imwrite(out_path, crop)
        print(f"[+] Saved zone {idx}: {out_path}")

        detected_zones.append((f"zone_{idx}", crop))
        idx += 1

    return detected_zones

def fix_common_errors(text):
    # Fix dash misread as م (common Tesseract mistake)
    import re
    text = re.sub(r'([ء-ي]+)م([ء-ي]+)', r'\1-\2', text)

    # Known replacements
    text = text.replace("موهران", "-وهران")
    text = text.replace("يصرى", "يسرى")
    text = text.replace("البيموهران", "البية-وهران")
    text = text.replace("البيةموهران", "البية-وهران")
    text = text.replace("الج-هورية", "الج-هورية")
    return text

# -------------------------------------------------------------
# OCR each zone with Tesseract (no preprocessing)
# -------------------------------------------------------------
# -------------------------------------------------------------
# OCR each zone with Tesseract (no preprocessing)
# -------------------------------------------------------------
def ocr_zone(img, zone_name):
    # Default padding
    padding = 12
    
    # Custom padding per zone
    if "zone_11" in zone_name:
        padding = 20  # Extra space for long authority text
    elif "zone_6" in zone_name or "zone_7" in zone_name:
        padding = 15  # Good clear space for Names
        
    # Add white padding to help Tesseract with edge text
    img = cv2.copyMakeBorder(
        img, padding, padding, padding, padding, 
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )
    txt = pytesseract.image_to_string(
        img,
        lang="fra+ara",
        config="--psm 7 --oem 3"
    ).strip()
    txt = fix_common_errors(txt)
    return txt


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    img = load_image(img_path)

    zones = detect_text_zones(img)

    results = []
    for name, crop in zones:
        txt = ocr_zone(crop, name)
        results.append({name: txt})

    with open("/app/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))