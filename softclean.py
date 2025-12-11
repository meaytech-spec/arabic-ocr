import torch
import cv2
import numpy as np
from softcleannet_model import SoftCleanNet


# --------------------------------------------------------
# Load SoftCleanNet (CPU only)
# --------------------------------------------------------
def load_softclean_model(path="/app/softcleannet.pth"):
    model = SoftCleanNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# --------------------------------------------------------
# AI Hologram Removal
# --------------------------------------------------------
def softclean_remove_hologram(img, model):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1]
    inp = gray.astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)

    # Inference
    with torch.no_grad():
        out = model(inp).squeeze().numpy()

    # Back to uint8
    cleaned = (out * 255).clip(0, 255).astype(np.uint8)

    # Post-process for smoothness and contrast
    cleaned = cv2.normalize(cleaned, None, 15, 235, cv2.NORM_MINMAX)
    cleaned = cv2.bilateralFilter(cleaned, 5, 45, 45)

    return cleaned


# --------------------------------------------------------
# Wrapper to load, clean, and save
# --------------------------------------------------------
def hologram_clean(input_path, output_path="softclean_output.jpg"):
    img = cv2.imread(input_path)

    if img is None:
        raise ValueError("❌ Cannot load input image")

    model = load_softclean_model("/app/softcleannet.pth")

    cleaned = softclean_remove_hologram(img, model)

    cv2.imwrite(output_path, cleaned)
    print("✔ SoftCleanNet hologram removal completed →", output_path)

    return cleaned
