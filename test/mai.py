from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import shutil
from detection_model import detector

app = FastAPI(
    title="铝尘检测API",
    description="支持图片和视频的铝尘检测服务",
    version="2.0.0"
)

# 创建必要的目录
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("uploaded_videos", exist_ok=True)
os.makedirs("processed_images", exist_ok=True)
os.makedirs("processed_videos", exist_ok=True)

# 挂载静态文件目录
app.mount("/processed_images", StaticFiles(directory="processed_images"), name="processed_images")
app.mount("/processed_videos", StaticFiles(directory="processed_videos"), name="processed_videos")

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

# 图片检测接口
@app.post("/detect/image/", response_model=DetectionResponse)
async def detect_aluminum_dust_image(file: UploadFile = File(...)):
    allowed_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_image_extensions:
        raise HTTPException(status_code=400, detail="不支持的图片格式")
    
    file_path = f"uploaded_images/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        detection_result = detector.detect(file_path)
        detection_result["annotated_image_url"] = f"/processed_images/annotated_{file.filename}"
        return detection_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片检测错误: {str(e)}")

# 视频检测接口
@app.post("/detect/video/", response_model=VideoDetectionResponse)
async def detect_aluminum_dust_video(file: UploadFile = File(...)):
    allowed_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_video_extensions:
        raise HTTPException(status_code=400, detail="不支持的视频格式")
    
    file_path = f"uploaded_videos/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        video_result = detector.detect_video(file_path)
        video_result["annotated_video_url"] = f"/processed_videos/annotated_{file.filename}"
        return video_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频检测错误: {str(e)}")

# 网页上传和显示界面
@app.get("/", response_class=HTMLResponse)
async def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>铝尘检测系统</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .result-section { margin: 20px 0; padding: 20px; border: 1px solid #4CAF50; border-radius: 8px; display: none; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .media-container { margin: 20px 0; text-align: center; }
            img, video { max-width: 100%; max-height: 500px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>铝尘检测系统</h1>
            
            <div class="upload-section">
                <h2>图片检测</h2>
                <form id="imageForm" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".jpg,.jpeg,.png,.bmp" required>
                    <button type="submit">上传图片检测</button>
                </form>
            </div>
            
            <div class="upload-section">
                <h2>视频检测</h2>
                <form id="videoForm" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".mp4,.avi,.mov,.mkv,.webm" required>
                    <button type="submit">上传视频检测</button>
                </form>
            </div>
            
            <div id="imageResult" class="result-section">
                <h3>图片检测结果</h3>
                <div id="imageStats"></div>
                <div class="media-container">
                    <img id="annotatedImage" src="" alt="标注后的图片">
                </div>
            </div>
            
            <div id="videoResult" class="result-section">
                <h3>视频检测结果</h3>
                <div id="videoStats"></div>
                <div class="media-container">
                    <video id="annotatedVideo" controls>
                        <source src="" type="video/mp4">
                        您的浏览器不支持视频播放
                    </video>
                </div>
            </div>
        </div>

        <script>
            // 图片上传处理
            document.getElementById('imageForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                
                try {
                    const response = await fetch('/detect/image/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        displayImageResult(result);
                    } else {
                        alert('上传失败');
                    }
                } catch (error) {
                    alert('发生错误: ' + error);
                }
            });
            
            // 视频上传处理
            document.getElementById('videoForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                
                try {
                    const response = await fetch('/detect/video/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        displayVideoResult(result);
                    } else {
                        alert('上传失败');
                    }
                } catch (error) {
                    alert('发生错误: ' + error);
                }
            });
            
            function displayImageResult(result) {
                document.getElementById('imageStats').innerHTML = `
                    <p>检测到 <strong>${result.detection_count}</strong> 个铝尘颗粒</p>
                    <p>平均置信度: <strong>${result.average_confidence || 0}</strong></p>
                `;
                document.getElementById('annotatedImage').src = result.annotated_image_url;
                document.getElementById('imageResult').style.display = 'block';
                document.getElementById('videoResult').style.display = 'none';
            }
            
            function displayVideoResult(result) {
                document.getElementById('videoStats').innerHTML = `
                    <p>总帧数: <strong>${result.total_frames}</strong></p>
                    <p>检测帧数: <strong>${result.analyzed_frames}</strong></p>
                    <p>总检测数量: <strong>${result.total_detections}</strong></p>
                    <p>平均每帧检测数: <strong>${result.average_detections_per_frame}</strong></p>
                    <p>视频时长: <strong>${result.video_duration}</strong> 秒</p>
                `;
                document.getElementById('annotatedVideo').src = result.annotated_video_url;
                document.getElementById('videoResult').style.display = 'block';
                document.getElementById('imageResult').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/")
async def api_info():
    return {
        "message": "铝尘检测API服务",
        "endpoints": {
            "图片检测": "POST /detect/image/",
            "视频检测": "POST /detect/video/",
            "网页界面": "GET /"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)