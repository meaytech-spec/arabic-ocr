import cv2
import numpy as np

# ----------------------------------------------------------
# 1) AUTO-ROTATE TO LANDSCAPE
# ----------------------------------------------------------
def auto_rotate(img):
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# ----------------------------------------------------------
# 2) PERSPECTIVE CORRECTION
# ----------------------------------------------------------
def detect_biggest_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[0]


def four_point_transform(img, pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH))


def correct_perspective(img):
    cnt = detect_biggest_contour(img)
    if cnt is None:
        return img

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

    if len(approx) == 4:
        return four_point_transform(img, approx)

    return img


# ----------------------------------------------------------
# 3) CAMSCANNER "MAGIC COLOR" FILTER
# ----------------------------------------------------------
def apply_camscanner_filter(img):
    """
    Simulates CamScanner's 'Magic Color' / Document Scan effect.
    1. Flattens illumination (removes shadows) using Division Normalization.
    2. Whitens background.
    3. Enhances text contrast.
    """
    # 1. Convert to Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # 2. Estimate Background (Illumination)
    # We use a large dilation to "erase" the text and keep the background
    # Kernel size depends on resolution, 25x25 is good for ~1000px width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    
    # Blur the background to smooth it out (removes artifacts from dilation)
    bg = cv2.GaussianBlur(bg, (21, 21), 0)
    
    # 3. Division Normalization
    # result = (gray / bg) * 255
    # This makes the background white (255) and preserves the text darkness relative to local bg
    # This is the "Magic" part that removes shadows.
    norm = cv2.divide(gray, bg, scale=255)
    
    # 4. Sharpening (Unsharp Masking)
    # Helps text pop out against the now-white background
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    sharpened = cv2.addWeighted(norm, 1.5, blur, -0.5, 0)
    
    # 5. Final Contrast Boost (CLAHE)
    # Enhances local contrast to make faint text readable
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final = clahe.apply(sharpened)
    
    return final


# ----------------------------------------------------------
# FINAL PIPELINE
# ----------------------------------------------------------
def preprocess_option_b(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image")

    # 1. Rotate
    img = auto_rotate(img)
    
    # 2. Perspective Correction
    img = correct_perspective(img)
    
    # 3. Apply CamScanner Magic Filter
    enhanced = apply_camscanner_filter(img)

    # Save debug version
    cv2.imwrite("/app/preprocessed_option_b.jpg", enhanced)

    return enhanced
