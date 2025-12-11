# CamScanner.py - Documentation

## Overview
This script replicates the CamScanner mobile app functionality for document scanning and enhancement.

## Available Modes

### 1. **clean** (Recommended for ID cards with holograms)
```bash
python camscanner.py input.jpg clean
```
**What it does:**
- Removes holograms completely
- Creates pure white background
- Only black text remains visible
- Matches CamScanner mobile app behavior

**Technical details:**
- Background normalization (removes shadows/holograms)
- Gamma correction (γ=1.3) for brightness
- Strong CLAHE (clipLimit=5.0) for maximum contrast
- Small block adaptive threshold (11x11) for aggressive whitening
- Morphological cleanup to remove noise

### 2. **magic** (Color enhancement)
```bash
python camscanner.py input.jpg magic
```
- Enhances colors while maintaining natural look
- CLAHE on LAB color space
- Good for colored documents

### 3. **bw** (Black & White)
```bash
python camscanner.py input.jpg bw
```
- Simple adaptive threshold
- Pure black and white output
- No background normalization

### 4. **gray** (Grayscale)
```bash
python camscanner.py input.jpg gray
```
- Converts to grayscale
- No enhancement

### 5. **sharp** (OCR Optimized)
```bash
python camscanner.py input.jpg sharp
```
- Grayscale with sharpening
- CLAHE enhancement
- Good for text recognition

### 6. **auto** (Default - Smart choice)
```bash
python camscanner.py input.jpg
# or
python camscanner.py input.jpg auto
```
- Automatically chooses between magic and sharp based on brightness
- If image is dark (brightness < 110): uses magic mode
- If image is bright: uses sharp mode

## Usage Examples

### Basic usage (auto mode):
```bash
docker run --rm -v "D:\dznotaire\dznotaire-app\ocr:/app" algerian-id-ocr python3 /app/camscanner.py /app/images/cni.jpg
```

### With specific mode:
```bash
docker run --rm -v "D:\dznotaire\dznotaire-app\ocr:/app" algerian-id-ocr python3 /app/camscanner.py /app/images/cni.jpg clean
```

### With custom output path:
```python
# Modify the script or use it programmatically:
from camscanner import camscanner
camscanner("/app/images/cni.jpg", mode="clean", save_path="/app/output.jpg")
```

## How CamScanner Mobile App Works

The mobile app uses these techniques (all implemented in 'clean' mode):

1. **Background Normalization**
   - Divides image by heavily blurred version
   - Removes shadows and uneven lighting
   - Eliminates hologram patterns

2. **Gamma Correction**
   - Brightens the entire image
   - Makes background lighter

3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Enhances local contrast
   - Makes text stand out more

4. **Adaptive Thresholding** (The Secret!)
   - Small block size (11x11 pixels)
   - High C value (15) for aggressive whitening
   - Converts everything to pure black or pure white
   - Background becomes 255 (white)
   - Text becomes 0 (black)

5. **Morphological Operations**
   - Removes tiny noise
   - Slightly bolds text for better readability

## Why 'clean' Mode is Best for ID Cards

ID cards have:
- ✅ Holographic security features
- ✅ Uneven lighting from camera flash
- ✅ Shadows and reflections
- ✅ Background patterns

The 'clean' mode:
- ✅ Removes ALL holograms
- ✅ Creates pure white background
- ✅ Makes text crisp and black
- ✅ Perfect for OCR processing

## Output

All modes save to `scan_result.jpg` by default.
The script prints: `[✓] Scanned using mode 'clean' → scan_result.jpg`
