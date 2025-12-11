FROM python:3.10-slim

WORKDIR /app

# -----------------------------------------------------
# Install system dependencies + Tesseract
# -----------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-fra \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Install PaddlePaddle + PaddleOCR
# -----------------------------------------------------
RUN pip install --upgrade pip
RUN pip install paddlepaddle==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install paddleocr==2.6.1.3 -i https://pypi.tuna.tsinghua.edu.cn/simple


# -----------------------------------------------------
# Install other dependencies
# -----------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless \
    flask \
    flask-cors \
    pytesseract \
    uvicorn[standard] \
    python-multipart \
    fastapi \
    imutils \
    Pillow \
    regex

# -----------------------------------------------------
# Copy OCR + model code
# -----------------------------------------------------
COPY softclean.py /app/softclean.py
COPY softcleannet_model.py /app/softcleannet_model.py
COPY softcleannet.pth /app/softcleannet.pth

COPY final_ocr_zones_fr.py /app/final_ocr_zones_fr.py
COPY server.py /app/server.py

EXPOSE 5005

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
