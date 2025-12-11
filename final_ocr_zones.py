import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import re
from datetime import datetime


# ============================================================
# PREPROCESSING (your original code, unchanged)
# ============================================================

def deskew(image):
    coords = np.column_stack(np.where(image < 255))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def auto_rotate(img):
    # Make sure image is landscape (width > height)
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def remove_hologram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2

    # Remove diagonal frequencies (hologram signature)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-5:ccol+5] = 0  # vertical stripe
    mask[crow-5:crow+5, ccol-30:ccol+30] = 0  # horizontal stripe

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # ⭐ Rotate portrait → landscape
    img = auto_rotate(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)

    denoised = cv2.bilateralFilter(gray, 11, 15, 15)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(denoised, -1, sharpen_kernel)

    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41, 15
    )

    # Do NOT deskew anymore
    processed = thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    # ⭐ Resize to stable dimensions (landscape only)
    morph = cv2.resize(morph, (1024, 612))

    cv2.imwrite("/app/preprocessed.jpg", morph)

    return morph, img


# ============================================================
# FIXED ZONES (corrected coordinates)
# ============================================================

ZONES = {
    # ---------- Authority / Issue / Expiry ----------
    "authority":   (550, 150, 990, 200),
    "issue_date":  (590, 190, 900, 230),
    "expiry_date": (590, 230, 900, 280),

    # ---------- National ID ----------
    "national_id": (440, 300, 730, 350),

    # ---------- Names ----------
    "last_name":   (600, 360, 960, 410),   # Widened
    "first_name":  (450, 410, 950, 470),   # Widened

    # ---------- Birth info line ----------
    "gender":      (400, 450, 650, 530),   # Widened significantly to catch value
    "birth_date":  (600, 460, 900, 510 ),   # تاريخ الميلاد

    # ---------- Birth place ----------
    "birth_place": (600, 510, 900, 550),
}

def enhance_last_name(img):
    """
    Aggressive enhancement specifically for last name field.
    Goal: Extract "عيواني" cleanly from hologram-interfered zone.
    """
    
    # 1️⃣ Convert to grayscale if needed
    if len(img.shape) == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()

    # 2️⃣ VERY AGGRESSIVE denoising (removes hologram patterns)
    g = cv2.bilateralFilter(g, 5, 75, 75)  # Strong bilateral filter
    
    # 3️⃣ Morphological opening to remove hologram specs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4️⃣ Strong upscaling for thin Arabic strokes
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 5️⃣ Additional denoising at larger size
    g = cv2.fastNlMeansDenoising(g, h=30, templateWindowSize=7, searchWindowSize=21)

    # 6️⃣ CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # 7️⃣ Adaptive threshold - tuned for clean Arabic extraction
    g = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,  # Larger block size for stable threshold
        15   # Higher C to handle varying illumination
    )

    # 8️⃣ Morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 9️⃣ Light dilation to strengthen characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    g = cv2.dilate(g, kernel, iterations=1)

    # Save for debugging
    cv2.imwrite("/app/debug_lastname_final.jpg", g)

    return g

def enhance_last_name_advanced(img):
    """
    Advanced hologram removal using:
    1. Median filtering (removes repeating patterns)
    2. Morphological reconstruction
    3. Connected component analysis
    4. Contour-based text extraction
    """
    
    # 1️⃣ Convert to grayscale if needed
    if len(img.shape) == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img.copy()

    # 2️⃣ Median blur - VERY effective for hologram patterns (repeating artifacts)
    g = cv2.medianBlur(g, 7)
    
    # 3️⃣ Morphological opening to remove small hologram noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4️⃣ Strong bilateral filter for edge-preserving smoothing
    g = cv2.bilateralFilter(g, 7, 80, 80)

    # 5️⃣ Upscale significantly
    g = cv2.resize(g, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # 6️⃣ Advanced denoising with non-local means
    g = cv2.fastNlMeansDenoising(g, h=35, templateWindowSize=7, searchWindowSize=21)

    # 7️⃣ CLAHE for uniform contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # 8️⃣ Invert to get white text on black (better for Tesseract)
    g = cv2.bitwise_not(g)

    # 9️⃣ Threshold with inverted binary
    g = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        61,  # Large block size
        20   # High C for aggressive foreground extraction
    )

    # 🔟 Morphological closing to connect character parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove small noise (hologram remnants)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save for debugging
    cv2.imwrite("/app/debug_lastname_final.jpg", g)

    return g


def ocr_zone(field, img):
    """
    OCR with field-specific optimizations.
    """

    # ============================================
    """
    OCR with field-specific optimizations.
    """

    # ============================================
    # Special treatment for last_name (high priority)
    # ============================================
    if field == "last_name":
        enhanced = enhance_last_name(img)

        # Try with Arabic + French first (highest confidence for names)
        txt = pytesseract.image_to_string(
            enhanced,
            lang="ara+fra",
            config="--psm 6 --oem 3 -c tessedit_char_whitelist=ابجدهوزحطيكلمنسعفصقرشتثخذضظغ"
        ).strip()

        # If that fails, try with more languages
        if not txt or len(txt) < 2:
            txt = pytesseract.image_to_string(
                enhanced,
                lang="ara+fra+eng",
                config="--psm 7 --oem 3"
            ).strip()

        # Last resort: basic grayscale without enhancement
        if not txt or len(txt) < 2:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            gray = cv2.resize(gray, None, fx=3, fy=3)
            txt = pytesseract.image_to_string(
                gray,
                lang="ara+fra",
                config="--psm 6 --oem 3"
            ).strip()

        return txt

    # ============================================
    # Optimized OCR for dates (numeric extraction)
    # ============================================
    elif field in ["birth_date", "issue_date", "expiry_date"]:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = cv2.resize(gray, None, fx=3, fy=3)
        
        # Enhance contrast for numbers
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Threshold optimized for digits
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )

        txt = pytesseract.image_to_string(
            gray,
            lang="0123456789",
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/.-"
        ).strip()

        return txt

    # ============================================
    # OCR for national ID (numeric)
    # ============================================
    elif field == "national_id":
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = cv2.resize(gray, None, fx=3, fy=3)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        txt = pytesseract.image_to_string(
            gray,
            lang="0123456789",
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
        ).strip()

        return txt

    # ============================================
    # OCR for Gender (Specific)
    # ============================================
    elif field == "gender":
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Upscale for better character recognition
        gray = cv2.resize(gray, None, fx=3, fy=3)
        
        # Light denoising
        gray = cv2.medianBlur(gray, 3)

        # Thresholding
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

        # Try Arabic + French + English with looser PSM
        txt = pytesseract.image_to_string(
            gray,
            lang="ara+fra+eng",
            config="--psm 6 --oem 3"
        ).strip()

        return txt

    # ============================================
    # Standard OCR for other text fields
    # ============================================
    else:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = cv2.resize(gray, None, fx=2, fy=2)

        txt = pytesseract.image_to_string(
            gray,
            lang="ara+fra+eng",
            config="--psm 6 --oem 3"
        ).strip()

        return txt

def ocr_zone(field, img):

    # ============================================
    # 1) Special treatment ONLY for last_name
    # ============================================
    if field == "last_name":
        enhanced = enhance_last_name(img)

        txt = pytesseract.image_to_string(
            enhanced,
            lang="ara+fra",
            config="--psm 7 --oem 3"
        ).strip()

        # If enhanced fails → fallback normal OCR
        if not txt or len(txt) < 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            txt = pytesseract.image_to_string(
                gray,
                lang="ara+fra+eng",
                config="--psm 7 --oem 3"
            ).strip()

        return txt

    # ============================================
    # 2) Default OCR for all other zones
    # ============================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    txt = pytesseract.image_to_string(
        gray,
        lang="ara+fra+eng",
        config="--psm 7 --oem 3"
    ).strip()

    return txt

def extract_zones_fixed(image):
    extracted = {}

    for field, (x1, y1, x2, y2) in ZONES.items():
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"/app/zone_{field}.jpg", crop)
        extracted[field] = crop

    return extracted

def ocr_all_zones(zone_images):
    results = {}
    for field, img in zone_images.items():
        results[field] = ocr_zone(field, img)
    return results




# ============================================================
# CLEANING + PARSING
# ============================================================

def clean_text(s):
    s = s.replace("\n", " ").strip()
    s = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s:/\-\.]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def remove_labels(text, field_type):
    if not text: return ""
    
    # Common labels to strip
    labels = []
    if field_type == "first_name":
        labels = [r"الإسم", r"الاسم", r"First Name", r"Name"]
    elif field_type == "last_name":
        labels = [r"اللقب", r"Last Name", r"Surname"]
    elif field_type == "birth_place":
        labels = [r"مكان الميلاد", r"مكان", r"الميلاد", r"Place of Birth"]
    elif field_type == "authority":
        labels = [r"سلطة الإصدار", r"سلطة", r"الإصدار", r"Authority"]
    elif field_type == "gender":
        labels = [r"الجنس", r"Sex", r"Gender"] 
        # Remove "09" or other digits if they appear as noise before the text
        text = re.sub(r"^\d+\s*", "", text)
    elif field_type == "national_id":
        labels = [r"رقم التعريف", r"National ID"]

    for label in labels:
        text = re.sub(rf"{label}[:\.\-]*", "", text)
    
    # Clean up leading/trailing punctuation
    text = re.sub(r"^[:\.\-\s]+", "", text)
    text = re.sub(r"[:\.\-\s]+$", "", text)
    
    return text.strip()


def convert_arabic_numbers(s):
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def extract_date(s):
    s = convert_arabic_numbers(s)
    # Allow dot, slash, hyphen, AND COMMA
    m = re.search(r"(\d{4})[./\-,](\d{2})[./\-,](\d{2})", s)
    if not m:
        return None
    y, mm, d = m.groups()
    try:
        dt = datetime(int(y), int(mm), int(d))
        return dt.strftime("%d/%m/%Y")
    except:
        return None


def clean_name(s):
    """
    Keeps only Arabic characters and spaces.
    Removes Latin characters, digits, and punctuation.
    """
    if not s:
        return ""
    # Keep only Arabic letters and whitespace
    s = re.sub(r"[^\u0600-\u06FF\s]", "", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ============================================================
# SMART PARSING USING ZONES FIRST
# ============================================================

def parse_id_from_zones(z):
    data = {
        "card_type": "بطاقة التعريف الوطنية",
        "republic": "الجمهورية الجزائرية الديمقراطية الشعبية",

        "national_id": z["national_id"],
        "last_name": z["last_name"],
        "first_name": z["first_name"],
        "gender": z["gender"],

        "birth_date": extract_date(z["birth_date"]),
        "birth_place": z["birth_place"],

        "issue_date": extract_date(z["issue_date"]),
        "expiry_date": extract_date(z["expiry_date"]),
        "authority": z["authority"],
    }

    # Clean Arabic text
    for k in ["last_name", "first_name", "birth_place", "authority", "gender", "national_id"]:
        if data[k]:
            data[k] = clean_text(data[k])
            data[k] = remove_labels(data[k], k)

            # Fix missing space after Ta Marbuta (ة)
            if k == "authority":
                data[k] = re.sub(r'(ة)([\u0600-\u06FF])', r'\1 \2', data[k])

    # Specific cleaning for names (remove digits/latin noise)
    for k in ["last_name", "first_name", "birth_place"]:
        if data[k]:
            data[k] = clean_name(data[k])

    return data



# ============================================================
# OUTPUT FORMATTER
# ============================================================

def format_output(d):
    out = []
    out.append("=" * 70)
    out.append("ALGERIAN ID CARD - EXTRACTED DATA")
    out.append("=" * 70)

    out.append(f"Card Type: {d['card_type']}\n")

    out.append("PERSONAL INFORMATION:")
    out.append("-" * 70)
    out.append(f"Last Name: {d['last_name']}")
    out.append(f"First Name: {d['first_name']}")
    out.append(f"National ID: {d['national_id']}")
    out.append(f"Gender: {d['gender']}")
    out.append(f"Birth Date: {d['birth_date']}")
    out.append(f"Birth Place: {d['birth_place']}")

    out.append("\nCARD INFORMATION:")
    out.append("-" * 70)
    out.append(f"Issuing Authority: {d['authority']}")
    out.append(f"Issue Date: {d['issue_date']}")
    out.append(f"Expiry Date: {d['expiry_date']}")

    out.append("=" * 70)
    return "\n".join(out)



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]

    processed, original = preprocess_image(img_path)

    # Resize original to fixed size for stable zones
    TARGET_W = 900
    TARGET_H = int(TARGET_W / 1.586)
    original = cv2.resize(original, (TARGET_W, TARGET_H))

    zones = extract_zones_fixed(original)

    # zone_text = ocr_all_zones(zones)
    # print("\n================ RAW OCR PER ZONE ================\n")
    # for field, text in zone_text.items():
    #     print(f"[{field}] => {repr(text)}\n")
    # print("=================================================\n")

    zone_text = ocr_all_zones(zones)
    data = parse_id_from_zones(zone_text)

    # Output strictly JSON for the server to parse
    import json
    # print(json.dumps(data, ensure_ascii=False))

    # Output strictly JSON
    result = json.dumps(data, ensure_ascii=False)

    # ⭐ Save using UTF-8 encoding (fixes Arabic corruption)
    with open("/app/result.json", "w", encoding="utf-8") as f:
        f.write(result)

    # Still print for Docker stdout
    print(result)
