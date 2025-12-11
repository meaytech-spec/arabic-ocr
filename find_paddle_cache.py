import os
from paddleocr import PaddleOCR

# Step 1: Force model download (if not downloaded yet)
print("🔄 Initializing PaddleOCR... (this will download the model)")

# Step 2: Search for .paddleocr folder
paths_to_check = [
    os.path.expanduser("~"),
    os.path.join(os.path.expanduser("~"), "AppData", "Local"),
    os.path.join(os.path.expanduser("~"), "AppData", "Roaming"),
    "C:\\Users",
    "C:\\",
]

found = False

print("\n🔍 Searching for .paddleocr folder...")

for base in paths_to_check:
    for root, dirs, files in os.walk(base):
        if ".paddleocr" in dirs:
            print("\n✅ FOUND! Folder path:")
            print(os.path.join(root, ".paddleocr"))
            found = True
            break
    if found:
        break

if not found:
    print("\n❌ Could not find .paddleocr. Try running OCR once manually.")
