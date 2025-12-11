import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import re
from datetime import datetime
from paddleocr import PaddleOCR


# -------------------------------------------------------------
# PaddleOCR 2.6 initialization (DETECTION ONLY)
# -------------------------------------------------------------
text_detector = PaddleOCR(
    lang="ch",        # IMPORTANT: Chinese model gives BEST detection
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


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = auto_rotate(img)

    # UPSCALE BEFORE ANYTHING (critical for detection)
    img = cv2.resize(img, None, fx=2.0, fy=2.0)

    return img, img.copy()


# -------------------------------------------------------------
# SUPER PREPROCESSOR (fix low contrast + background pattern)
# -------------------------------------------------------------
def extract_blue(img):
    b, g, r = cv2.split(img)
    return b

def retinex_enhance(img):
    img_float = img.astype(np.float32) + 1.0
    log_img = np.log(img_float)
    blur = cv2.GaussianBlur(log_img, (51, 51), 0)
    retinex = log_img - blur
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def top_hat(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def sharpen(img):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def super_enhance_detection(img):
    # Convert to LAB for better contrast handling
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE ONLY on the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge back
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Convert to gray
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Slight sharpening (NOT heavy)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    # Upscale
    big = cv2.resize(sharp, None, fx=2.0, fy=2.0)

    cv2.imwrite("/app/debug_super_preprocessed.jpg", big)

    return big


# -------------------------------------------------------------
# Draw detected boxes
# -------------------------------------------------------------
def draw_detected_boxes(img, ocr_result, save_path="/app/detected_boxes.jpg"):
    img = img.copy()

    # Handle PaddleOCR output structure [[box1, box2, ...]]
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
# Filter small/noisy boxes
# -------------------------------------------------------------
def is_text_zone(w, h):
    if h < 12 or w < 40:
        return False
    return True


# -------------------------------------------------------------
# DETECT TEXT ZONES (Corrected + SUPER PREPROCESSING)
# -------------------------------------------------------------
def detect_text_zones(original_img):

    detected_zones = []

    # SUPER PREPROCESSING
    enhanced = super_enhance_detection(original_img)
    cv2.imwrite("/app/_enhanced_input.jpg", enhanced)

    # IMPORTANT
    ocr_result = text_detector.ocr("/app/_enhanced_input.jpg", det=True, rec=False, cls=False)

    print("\n=== RAW DETECTION ===")
    print(ocr_result)

    draw_detected_boxes(enhanced, ocr_result)  # draw on enhanced

    # compute scale ratio between enhanced and original
    scale_x = enhanced.shape[1] / original_img.shape[1]
    scale_y = enhanced.shape[0] / original_img.shape[0]

    idx = 1

    # Get boxes from the first image result
    boxes = ocr_result[0] if ocr_result else []

    for box in boxes:
        # box is the list of points [[x,y], ...]
        pts = []

        for pt in box:
            if isinstance(pt, list) and len(pt) == 2:
                pts.append([int(pt[0]), int(pt[1])])

        if len(pts) != 4:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # coordinates on enhanced image
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # convert coordinates back to original image
        xo1 = int(x1 / scale_x)
        yo1 = int(y1 / scale_y)
        xo2 = int(x2 / scale_x)
        yo2 = int(y2 / scale_y)

        crop = original_img[yo1:yo2, xo1:xo2]
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


# -------------------------------------------------------------
# French OCR Enhancement (Tesseract)
# -------------------------------------------------------------
def enhance_french(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    gray = cv2.medianBlur(gray, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# -------------------------------------------------------------
# OCR each zone with Tesseract
# -------------------------------------------------------------
def ocr_zone(img):
    enhanced = enhance_french(img)
    txt = pytesseract.image_to_string(
        enhanced,
        lang="fra+ara",
        config="--psm 7 --oem 3"
    ).strip()
    return txt


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    _, original = preprocess_image(img_path)

    zones = detect_text_zones(original)

    results = []
    for name, crop in zones:
        txt = ocr_zone(crop)
        results.append({name: txt})

    with open("/app/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
