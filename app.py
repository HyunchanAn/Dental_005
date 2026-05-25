import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Caries Detection API")

# 전역 컨텍스트에 모델 로드 (Inference 속도 최적화)
# 여기서는 9클래스 960px 모델을 기본으로 사용합니다.
model = YOLO("yolov8x_AlphaDent_9_classes_960px.pt")

@app.post("/predict")
async def predict_caries(file: UploadFile = File(...)):
    # 1. 이미지 로드 및 전처리
    request_object_content = await file.read()
    img_array = np.frombuffer(request_object_content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 2. YOLOv8 Instance Segmentation 추론
    results = model(img, imgsz=960, conf=0.25)
    
    # 3. 마킹 오버레이 이미지 생성
    annotated_img = results[0].plot()
    
    # 4. 이미지 스트림 반환
    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
