from paddleocr import PaddleOCR
import os


# Print where models are stored
print("\nModel loaded. Now checking model paths...\n")

root = os.path.expanduser("~")
print("HOME DIR:", root)

paddle_dir = os.path.join(root, ".paddleocr")
print("PADDLE DIR:", paddle_dir)

if os.path.exists(paddle_dir):
    print("FOUND:", paddle_dir)
    for r, d, f in os.walk(paddle_dir):
        print(" ", r)
else:
    print("❌ .paddleocr folder NOT found under home directory.")
