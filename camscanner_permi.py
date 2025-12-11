import cv2
import numpy as np

def detect_text(image):
    """
    Text detection placeholder.
    You can plug Tesseract, EasyOCR, PaddleOCR, or Qari-OCR here.
    """
    # Example: convert to grayscale (optional for OCR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray   # return the image prepared for OCR


if __name__ == "__main__":
    img = cv2.imread("permi.jpg")
    
    if img is None:
        print("❌ Cannot load image.")
        exit()

    # Only text detection (no treatment)
    text_ready = detect_text(img)

    # Save the detection-ready version
    cv2.imwrite("permi_text_ready.jpg", text_ready)
    print("✔ Text detection image ready → permi_text_ready.jpg")
