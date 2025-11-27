from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import shutil
import traceback  # 添加这个导入
from detection_model import detector
import uvicorn

app = FastAPI(
    title="Aluminum Dust Detection API",
    version="2.0"
)

os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("uploaded_videos", exist_ok=True)

app.mount("/uploaded_images", StaticFiles(directory="uploaded_images"), name="uploaded_images")
app.mount("/uploaded_videos", StaticFiles(directory="uploaded_videos"), name="uploaded_videos")

class DetectionResponse(BaseModel):
    detection_count: int
    bboxes: List[List[float]]
    confidences: List[float]
    detections: List[Dict[str, Any]]
    annotated_image_url: str

class VideoDetectionResponse(BaseModel):
    total_frames: int
    total_detections: int
    average_detections_per_frame: float
    frames_summary: List[Dict[str, Any]]
    annotated_video_url: str

# Image Detection Interface
@app.post("/detect/image/", response_model=DetectionResponse)
async def detect_aluminum_dust_image(file: UploadFile = File(...)):
    allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_image_extensions:
        raise HTTPException(status_code=400, detail="Unsupport")
    
    # Save file
    original_path = f"uploaded_images/{file.filename}"
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print(f"Starting detection for: {original_path}")  # 添加日志
        detection_result = detector.detect(original_path)
        detection_result["annotated_image_url"] = f"/uploaded_images/annotated_{file.filename}"
        print(f"Detection completed: {detection_result}")  # 添加日志
        return detection_result
    except Exception as e:
        # 打印详细的错误信息
        error_traceback = traceback.format_exc()
        print(f"Error during detection: {str(e)}")
        print(f"Full traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

# Video Detection Interface
@app.post("/detect/video/", response_model=VideoDetectionResponse)
async def detect_aluminum_dust_video(file: UploadFile = File(...)):
    allowed_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_video_extensions:
        raise HTTPException(status_code=400, detail="Unsupport")
    
    original_path = f"uploaded_videos/{file.filename}"
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        print(f"Starting video detection for: {original_path}")
        video_result = detector.detect_video(original_path)
        video_result["annotated_video_url"] = f"/uploaded_videos/annotated_{file.filename}"
        print(f"Video detection completed")
        return video_result
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error during video detection: {str(e)}")
        print(f"Full traceback: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

@app.get("/api/")
async def api_info():
    return {
        "message": "Aluminum dust detection API",
        "endpoints": {
            "Image detection": "POST /detect/image/",
            "Video detection": "POST /detect/video/",
            "API documentation": "GET /docs"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Aluminum Dust Detection API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)