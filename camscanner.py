import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def upscale_and_clarify(input_path, output_path):
    """
    High-quality enhancement for ID cards / documents:
    ✔ Upscale (critical for OCR)
    ✔ Shadow removal
    ✔ Text clarity boost
    ✔ Non-destructive contrast
    ✔ Ultra-gentle sharpening (no bold text)
    """

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load {input_path}")

    # ------------------------
    # 1) UPSCALING (x2)
    # ------------------------
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # ------------------------
    # 2) VERY NATURAL DENOISING
    # ------------------------
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

    # ------------------------
    # 3) SHADOW REMOVAL (Stable version)
    # ------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BIG blur to estimate lighting, NOT morphology (cleaner result)
    bg = cv2.GaussianBlur(gray, (101, 101), 0)

    # Avoid division artifacts
    bg = cv2.add(bg, 1)

    # Light normalization
    shadow_free = cv2.divide(gray, bg, scale=255)
    shadow_free = np.uint8(cv2.normalize(shadow_free, None, 0, 255, cv2.NORM_MINMAX))

    # ------------------------
    # 4) MERGE shadow-free luminance back into color image
    # ------------------------
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(result)

    # Blend 70% original + 30% shadow-free to avoid over-flattening
    L = cv2.addWeighted(L, 0.7, shadow_free, 0.3, 0)

    result = cv2.merge((L, A, B))
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    # ------------------------
    # 5) TEXT CLARITY BOOST (Gamma < 1 brightens text)
    # ------------------------
    gamma = 0.8
    inv = 1.0 / gamma
    table = (np.array([(i / 255.0) ** inv * 255 for i in range(256)])).astype("uint8")
    result = cv2.LUT(result, table)

    # ------------------------
    # 6) SOFT SHARPENING (NON-BOLD)
    # ------------------------
    sharpen_kernel = np.array([
        [0, -0.05, 0],
        [-0.05, 1.15, -0.05],
        [0, -0.05, 0]
    ])
    result = cv2.filter2D(result, -1, sharpen_kernel)

    # ------------------------
    # 7) SAVE HIGH QUALITY
    # ------------------------
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"✅ Enhanced scan saved to: {output_path}")

    return result



if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="High Quality Scanner-Style Enhancement")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", default=None, help="Output image path")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    # Auto output name
    if args.output:
        out = args.output
    else:
        name, ext = os.path.splitext(args.input)
        out = f"{name}_HQscan{ext}"

    upscale_and_clarify(args.input, out)
