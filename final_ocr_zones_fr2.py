import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
from paddleocr import PaddleOCR


# -------------------------------------------------------------
# Initialize PaddleOCR (Detection Only)
# -------------------------------------------------------------
text_detector = PaddleOCR(
    lang="ch",        # Chinese model = best box detection
    use_gpu=False,
    show_log=False
)


# -------------------------------------------------------------
# AUTO ROTATE + UPSCALE for better detection consistency
# -------------------------------------------------------------
def prepare_for_detection(img):
    h, w = img.shape[:2]

    # Auto rotate portrait images
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Upscale x1.5 to stabilize box detection
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return img


# -------------------------------------------------------------
# OCR-ONLY PREPROCESSING (Detection stays RAW)
# -------------------------------------------------------------
def preprocess_for_ocr(crop):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 2. Histogram equalization (light)
    gray = cv2.equalizeHist(gray)

    # 3. Gentle denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    # 4. Adaptive threshold (Arabic + French friendly)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    return th


# -------------------------------------------------------------
# Draw detected boxes for debugging
# -------------------------------------------------------------
def draw_detected_boxes(img, ocr_result, save_path="/app/detected_boxes_final.jpg"):
    img = img.copy()
    boxes = ocr_result[0] if ocr_result else []

    for idx, box in enumerate(boxes):
        pts = np.array(box).astype(int)
        cv2.polylines(img, [pts], True, (0,255,0), 3)
        x, y = pts[0]
        cv2.putText(img, str(idx+1), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imwrite(save_path, img)
    print(f"[+] Saved detection preview → {save_path}")


# -------------------------------------------------------------
# Detect and extract text zones
# -------------------------------------------------------------
def detect_text_zones(original_img):
    img = prepare_for_detection(original_img)
    temp_path = "/app/_temp_detect.jpg"
    cv2.imwrite(temp_path, img)

    ocr_result = text_detector.ocr(temp_path, det=True, rec=False, cls=False)
    draw_detected_boxes(img, ocr_result)

    boxes = ocr_result[0] if ocr_result else []
    detected = []

    idx = 1
    for box in boxes:
        pts = np.array(box).astype(int)

        x1, y1 = pts[:,0].min(), pts[:,1].min()
        x2, y2 = pts[:,0].max(), pts[:,1].max()

        crop = img[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        h, w = crop.shape[:2]
        if h < 15 or w < 40:  # stricter filtering
            continue

        detected.append((f"zone_{idx}", crop))
        idx += 1

    return detected


# -------------------------------------------------------------
# OCR with Tesseract on the enhanced crop
# -------------------------------------------------------------
def ocr_zone(img):
    cleaned = preprocess_for_ocr(img)

    text = pytesseract.image_to_string(
        cleaned,
        lang="ara+fra",
        config="--oem 3 --psm 6"
    )

    return text.strip()


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    original = cv2.imread(img_path)

    if original is None:
        raise ValueError("❌ Cannot load image")

    zones = detect_text_zones(original)

    results = []
    for name, crop in zones:
        txt = ocr_zone(crop)
        results.append({name: txt})

    with open("/app/result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
