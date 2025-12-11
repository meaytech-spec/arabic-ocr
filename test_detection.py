from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_gpu=False,
    show_log=True
)

img_path = "/app/cni.jpg"

result = ocr.ocr(img_path, det=True, rec=False, cls=False)

print("\nDetection result:")
for line in result:
    print(line[0])
