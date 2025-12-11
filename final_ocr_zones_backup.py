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


def preprocess_image(image_path):
    img = cv2.imread(image_path)

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
        cv2.THRESH_BINARY, 41, 15  # Increased block size and C
    )

    deskewed = deskew(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(deskewed, cv2.MORPH_CLOSE, kernel)
    TARGET_W = 1024
    TARGET_H = 612
    morph = cv2.resize(morph, (TARGET_W, TARGET_H))
    cv2.imwrite("/app/preprocessed.jpg", morph)
    return morph, img



# ============================================================
# FIXED ZONES (corrected coordinates)
# ============================================================

ZONES = {
    # ---------- Authority / Issue / Expiry ----------
    "authority":   (590, 120, 980, 190),
    "issue_date":  (590, 180, 900, 230),
    "expiry_date": (590, 230, 900, 280),

    # ---------- National ID ----------
    "national_id": (450, 300, 720, 350),

    # ---------- Names (wider capture area) ----------
    "last_name":   (450, 345, 900, 405),   # Wider  
    "first_name":  (450, 410, 900, 465),   # Wider

    # ---------- Birth info line ----------
    "gender":      (440, 480, 640, 525),   # الجنس
    "birth_date":  (600, 480, 900, 510 ),   # تاريخ الميلاد

    # ---------- Birth place ----------
    "birth_place": (600, 500, 900, 550),
}




# ============================================================
# ZONE OCR
# ============================================================

def extract_zones_fixed(image):
    extracted = {}

    for field, (x1, y1, x2, y2) in ZONES.items():
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"/app/zone_{field}.jpg", crop)
        extracted[field] = crop

    return extracted


def ocr_zone(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    txt = pytesseract.image_to_string(gray, lang="ara+fra+eng", config="--psm 7 --oem 3")
    
    return txt.strip()

def ocr_zone_name(img):
    """Specialized OCR for name fields using PSM 6 for better text capture"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # More aggressive resize for names
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5)
    txt = pytesseract.image_to_string(gray, lang="ara+eng", config="--psm 6 --oem 3")
    return txt.strip()




def ocr_all_zones(zone_images):
    results = {}
    for field, img in zone_images.items():
        # Use specialized function for names
        if field in ["last_name", "first_name"]:
            results[field] = ocr_zone_name(img)
        else:
            results[field] = ocr_zone(img)
    return results


def ocr_full_page(image):
    """Perform full-page OCR for pattern-based extraction"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize for better OCR
    gray = cv2.resize(gray, None, fx=2, fy=2)
    # Use PSM 4 for single column of text
    text = pytesseract.image_to_string(gray, lang="ara+fra+eng", config="--psm 4 --oem 3")
    return text


def extract_names_from_full_text(text):
    """Extract names using regex pattern matching"""
    text = clean_text(text)
    
    last_name = ""
    first_name = ""
    
    # Pattern 1: Find text after "اللقب" (Last Name label)
    # Look for: اللقب followed by : or whitespace, then capture text until next label/number
    lname_pattern = r"(?:اللقب|اللتب|اللفب)[:\s]+([^\d\n]+?)(?=\s*(?:الإسم|الاسم|رقم|\d{10,}|$))"
    lname_match = re.search(lname_pattern, text, re.UNICODE)
    if lname_match:
        last_name = lname_match.group(1).strip()
    
    # Pattern 2: Find text after "الإسم" or "الاسم" (First Name label)
    fname_pattern = r"(?:الإسم|الاسم)[:\s]+([^\d\n]+?)(?=\s*(?:الجنس|تاريخ|مكان|\d{10,}|$))"
    fname_match = re.search(fname_pattern, text, re.UNICODE)
    if fname_match:
        first_name = fname_match.group(1).strip()
    
    # Clean extracted names
    if last_name:
        last_name = clean_name(last_name)
        if not is_valid_arabic_text(last_name):
            last_name = ""
    
    if first_name:
        first_name = clean_name(first_name)
        if not is_valid_arabic_text(first_name):
            first_name = ""
    
    return {
        "last_name": last_name,
        "first_name": first_name
    }



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
        labels = [r"الجنس", r"Sex", r"Gender", r"01", r"02"] # Also strip codes
    elif field_type == "national_id":
        labels = [r"رقم التعريف", r"National ID"]

    for label in labels:
        text = re.sub(rf"{label}[:\.\-]*", "", text)
    
    # Clean up leading/trailing punctuation
    text = re.sub(r"^[:\.\-\s]+", "", text)
    text = re.sub(r"[:\.\-\s]+$", "", text)
    
    return text.strip()


def clean_name(text):
    """Remove OCR noise from name fields"""
    if not text:
        return ""
    
    # Remove digits
    text = re.sub(r"\d+", "", text)
    
    # If has Arabic, remove all Latin characters
    if re.search(r"[\u0600-\u06FF]", text):
        text = re.sub(r"[A-Za-z]", "", text)
    
    # Remove special characters and directional marks
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # If result is too short or looks like garbage, return empty
    if len(text) < 2 or len(text) > 50:
        return ""
    
    return text


def is_valid_arabic_text(text):
    """Check if text looks like valid Arabic (not OCR garbage)"""
    if not text or len(text) < 2:
        return False
    
    # Count Arabic characters
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    
    # If mixed with too much Latin, it's likely garbage
    if latin_chars > arabic_chars:
        return False
    
    # Check for excessive special characters (noise indicators)
    special_chars = len(re.findall(r"[^\w\s\u0600-\u06FF]", text))
    if special_chars > len(text) // 3:  # More than 1/3 special chars
        return False
    
    return True


def convert_arabic_numbers(s):
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def extract_date(s):
    s = convert_arabic_numbers(s)
    m = re.search(r"(\d{4})[./\-](\d{2})[./\-](\d{2})", s)
    if not m:
        return None
    y, mm, d = m.groups()
    try:
        dt = datetime(int(y), int(mm), int(d))
        return dt.strftime("%d/%m/%Y")
    except:
        return None



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

    # Specific cleaning for names (remove digits/latin noise)
    for k in ["last_name", "first_name", "birth_place"]:
        if data[k]:
            # Validate before cleaning
            if not is_valid_arabic_text(data[k]):
                data[k] = ""  # Clear garbage
                continue
            data[k] = clean_name(data[k])

    return data


def parse_id_hybrid(zone_text, full_page_text):
    """Parse ID using zones + full-page OCR fallback for names"""
    # Start with zone-based extraction
    data = parse_id_from_zones(zone_text)
    
    # If names are missing, try full-page extraction
    if not data["last_name"] or not data["first_name"]:
        names = extract_names_from_full_text(full_page_text)
        
        if not data["last_name"] and names["last_name"]:
            data["last_name"] = names["last_name"]
        
        if not data["first_name"] and names["first_name"]:
            data["first_name"] = names["first_name"]
    
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

    zone_text = ocr_all_zones(zones)
    
    # Perform full-page OCR for name extraction fallback
    full_page_text = ocr_full_page(original)
    
    print("\n================ FULL PAGE OCR ================\n")
    print(full_page_text[:500])  # Show first 500 chars
    print("\n===========================================\n")
    
    print("\n================ RAW OCR PER ZONE ================\n")
    for field, text in zone_text.items():
        print(f"[{field}] => {repr(text)}\n")
    print("=================================================\n")

    # Use hybrid approach: zones + full-page fallback
    data = parse_id_hybrid(zone_text, full_page_text)

    print(format_output(data))
