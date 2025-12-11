import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import re
from datetime import datetime
from preprocess_option_b import preprocess_option_b   # OPTION B PREPROCESSING


# ============================================================
# FIXED ZONES (coordinates corrected for 900×568 layout)
# ============================================================

ZONES = {
    # ---------- Authority / Issue / Expiry ----------
    "authority":   (550, 150, 990, 200),
    "issue_date":  (590, 190, 900, 230),
    "expiry_date": (590, 230, 900, 280),

    # ---------- National ID ----------
    "national_id": (450, 300, 720, 350),

    # ---------- Names ----------
    "last_name":   (600, 360, 960, 410),   # Widened
    "first_name":  (450, 410, 950, 470),   # Widened

    # ---------- Birth info line ----------
    "gender":      (400, 450, 650, 530),   # Widened significantly to catch value
    "birth_date":  (600, 460, 900, 510 ),   # تاريخ الميلاد

    # ---------- Birth place ----------
    "birth_place": (600, 510, 900, 550),
}


def ensure_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ============================================================
# SPECIAL ENHANCEMENT FOR LAST NAME (Option B aggressive)
# ============================================================

def enhance_last_name(img):

    gray = ensure_gray(img)

    # Median blur removes hologram lines
    gray = cv2.medianBlur(gray, 7)

    # Morph open removes small hologram dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Bilateral filter preserves Arabic strokes
    # Bilateral filter works on 8-bit 1-channel or 3-channel images
    gray = cv2.bilateralFilter(gray, 7, 80, 80)

    # Upscale strongly
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Non-local means denoising
    gray = cv2.fastNlMeansDenoising(gray, h=35)

    # CLAHE equalization
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Threshold (optimized for Arabic)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        61, 20
    )

    # Connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite("/app/debug_last_name.jpg", gray)
    return gray


# ============================================================
# MAIN OCR PER FIELD (MODE 1 = FULL ADVANCED)
# ============================================================

def ocr_zone(field, img):

    # ---------------------------
    # LAST NAME → Super enhanced
    # ---------------------------
    if field == "last_name":
        enhanced = enhance_last_name(img)

        txt = pytesseract.image_to_string(
            enhanced,
            lang="ara+fra",
            config="--psm 6 --oem 3"
        ).strip()

        if not txt or len(txt) < 2:
            gray = ensure_gray(img)
            gray = cv2.resize(gray, None, fx=3, fy=3)
            txt = pytesseract.image_to_string(
                gray, lang="ara+fra+eng", config="--psm 7"
            ).strip()

        return txt

    # ---------------------------
    # DATES → Strict numeric OCR
    # ---------------------------
    if field in ["birth_date", "issue_date", "expiry_date"]:
        gray = ensure_gray(img)
        gray = cv2.resize(gray, None, fx=3, fy=3)

        clahe = cv2.createCLAHE(clipLimit=2.0)
        gray = clahe.apply(gray)

        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )

        return pytesseract.image_to_string(
            gray,
            lang="eng",
            config="--psm 7 -c tessedit_char_whitelist=0123456789/.-"
        ).strip()

    # ---------------------------
    # NATIONAL ID → Numbers only
    # ---------------------------
    if field == "national_id":
        gray = ensure_gray(img)
        gray = cv2.resize(gray, None, fx=3, fy=3)

        return pytesseract.image_to_string(
            gray,
            lang="eng",
            config="--psm 7 -c tessedit_char_whitelist=0123456789"
        ).strip()

    # ---------------------------
    # GENDER → Arabic/French/English
    # ---------------------------
    if field == "gender":
        gray = ensure_gray(img)
        gray = cv2.resize(gray, None, fx=3, fy=3)
        gray = cv2.medianBlur(gray, 3)

        return pytesseract.image_to_string(
            gray,
            lang="ara+fra+eng",
            config="--psm 6"
        ).strip()

    # ---------------------------
    # DEFAULT OCR
    # ---------------------------
    gray = ensure_gray(img)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    return pytesseract.image_to_string(
        gray,
        lang="ara+fra+eng",
        config="--psm 6"
    ).strip()


# ============================================================
# ZONE EXTRACTION
# ============================================================

def extract_zones(image):
    zones = {}
    for field, (x1, y1, x2, y2) in ZONES.items():
        # Ensure coordinates are within bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"/app/zone_{field}.jpg", crop)
        zones[field] = crop
    return zones


# ============================================================
# CLEANING + PARSING
# ============================================================

def convert_ar_digits(s):
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

def extract_date(s):
    s = convert_ar_digits(s)
    m = re.search(r"(\d{4})[./-](\d{2})[./-](\d{2})", s)
    if not m:
        return None
    y, mm, d = m.groups()
    return f"{d}/{mm}/{y}"

def clean_arabic(s):
    return re.sub(r"[^\u0600-\u06FF\s]", "", s).strip()


def parse_data(z):

    return {
        "card_type": "بطاقة التعريف الوطنية",
        "republic": "الجمهورية الجزائرية الديمقراطية الشعبية",

        "national_id": convert_ar_digits(z["national_id"]),
        "last_name": clean_arabic(z["last_name"]),
        "first_name": clean_arabic(z["first_name"]),
        "gender": clean_arabic(z["gender"]).replace("الجنس", "").strip(),

        "birth_date": extract_date(z["birth_date"]),
        "birth_place": clean_arabic(z["birth_place"]).replace("مكان الميلاد", "").strip(),

        "issue_date": extract_date(z["issue_date"]),
        "expiry_date": extract_date(z["expiry_date"]),
        "authority": clean_arabic(z["authority"]).replace("سلطة الإصدار", "").strip(),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]

    # OPTION B PREPROCESSING
    processed = preprocess_option_b(img_path)

    # Resize to 900px width (zone coordinates calibrated)
    original = cv2.resize(processed, (900, 568))

    zones = extract_zones(original)

    zone_text = {k: ocr_zone(k, v) for k, v in zones.items()}

    data = parse_data(zone_text)

    # Save JSON (UTF-8 correct for Arabic)
    with open("/app/result.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))

    print(json.dumps(data, ensure_ascii=False))
