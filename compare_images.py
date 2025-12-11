import cv2
import numpy as np

def analyze(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load {path}")
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Brightness
    mean_val = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    # Background analysis (Peak of histogram)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peak_val = np.argmax(hist)
    
    return {
        "res": (w, h),
        "mean_brightness": mean_val,
        "contrast": contrast,
        "background_peak": peak_val,
        "hist": hist
    }

cni1 = analyze("/app/images/cni.jpg")
cni2 = analyze("/app/images/cni2.jpg")

print(f"--- CNI 1 (Original) ---")
print(f"Resolution: {cni1['res']}")
print(f"Brightness: {cni1['mean_brightness']:.2f}")
print(f"Contrast:   {cni1['contrast']:.2f}")
print(f"BG Peak:    {cni1['background_peak']}")

print(f"\n--- CNI 2 (Target) ---")
print(f"Resolution: {cni2['res']}")
print(f"Brightness: {cni2['mean_brightness']:.2f}")
print(f"Contrast:   {cni2['contrast']:.2f}")
print(f"BG Peak:    {cni2['background_peak']}")

# Calc ratio
res_ratio = cni2['res'][0] / cni1['res'][0]
print(f"\nResolution Ratio (cni2/cni1): {res_ratio:.2f}x")
