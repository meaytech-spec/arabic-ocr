from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import json
import cv2
from final_ocr_zones_fr import load_image, detect_text_zones, ocr_zone

app = FastAPI(
    title="DZ Notary OCR API",
    description="OCR Zones Extraction for Algerian IDs (CNI / Permis / Passport)",
    version="1.0.0"
)

# -------------------------------------------------------------
# CORS (allow mobile apps & browsers)
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------------------------------------------------
# TEST ROUTE
# -------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "DZ OCR API is running", "status": "ok"}


# -------------------------------------------------------------
# MAIN OCR ENDPOINT
# -------------------------------------------------------------
@app.post("/ocr")
async def ocr_process(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        temp_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load image
        img = load_image(temp_path)
        if img is None:
            return {"error": "Invalid image format"}

        # Detect zones
        zones = detect_text_zones(img)

        # OCR each zone
        results = []
        for name, crop in zones:
            text = ocr_zone(crop, name)
            results.append({name: text})

        # Output result.json
        json_path = "/app/result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "ocr_result": results
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# -------------------------------------------------------------
# Run in Docker
# -------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5005)
