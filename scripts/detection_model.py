from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import List, Dict, Any
import pygame
import threading  # 添加线程支持

class AluminumDustDetector:
    def __init__(self, model_path: str = "model/best.pt"):
        self.model_path = model_path
        self.model = None
        self.track_history = {}
        self.model_loaded = False
        self.model_error = None
        self.audio_initialized = False
        self.audio_file = self.find_audio_file()
    
    def find_audio_file(self):
        """查找音频文件位置"""
        possible_paths = [
            "sound_alert.wav",  # 项目根目录
            "../sound_alert.wav",
            "scripts/sound_alert.wav",
            "../scripts/sound_alert.wav",
            "uploaded_videos/sound_alert.wav"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found audio file at: {path}")
                return path
        
        print("Audio file not found, sound alerts will be disabled")
        return None
    
    def init_audio(self):
        """初始化音频系统"""
        if not self.audio_initialized and self.audio_file:
            try:
                pygame.mixer.init()
                self.audio_initialized = True
                print("Audio system initialized successfully!")
            except Exception as e:
                print(f"Audio initialization failed: {e}")
                self.audio_initialized = False
    
    def play_detection_sound_async(self):
        """异步播放音频（不阻塞主线程）"""
        def play_sound():
            try:
                if not self.audio_file or not os.path.exists(self.audio_file):
                    print("Audio file not available, skipping sound")
                    return
                
                self.init_audio()
                if not self.audio_initialized:
                    print("Audio system not available")
                    return
                
                # 设置音量并播放
                pygame.mixer.music.set_volume(0.7)  # 70% 音量
                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.play()
                print("🔊 Detection start sound played!")
                
                # 等待播放完成
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            except Exception as e:
                print(f"Error playing sound: {e}")
        
        # 在新线程中播放音频
        sound_thread = threading.Thread(target=play_sound)
        sound_thread.daemon = True  # 设置为守护线程
        sound_thread.start()
    
    def play_detection_sound(self):
        """播放检测开始音频"""
        if self.audio_file and os.path.exists(self.audio_file):
            self.play_detection_sound_async()
        else:
            print("🔊 Detection started (sound file not found)")
    
    def load_model(self):
        """延迟加载模型"""
        if self.model is None and not self.model_loaded:
            try:
                print(f"Loading model from: {self.model_path}")
                
                # 检查模型文件是否存在
                if not os.path.exists(self.model_path):
                    alternative_paths = [
                        "../model/best.pt",
                        "scripts/model/best.pt", 
                        "/app/scripts/model/best.pt"
                    ]
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            self.model_path = alt_path
                            print(f"Found model at: {alt_path}")
                            break
                    else:
                        raise FileNotFoundError(f"Model file not found at any expected location")
                
                self.model = YOLO(self.model_path)
                self.model_loaded = True
                print("Model loaded successfully!")
                
                # 预加载音频系统
                self.init_audio()
                
            except Exception as e:
                self.model_error = str(e)
                print(f"Error loading model: {e}")
                self.model = None
                self.model_loaded = False
    
    def detect(self, image_path: str) -> Dict:
        """Detection method using tracking"""
        # 播放检测开始音频
        self.play_detection_sound()
        
        self.load_model()  # 确保模型已加载
        
        if self.model is None:
            return {
                "error": f"Model not available: {self.model_error}",
                "detection_count": 0,
                "bboxes": [],
                "confidences": [],
                "track_ids": [],
                "detections": [],
                "annotated_image_path": image_path,
                "average_confidence": 0.0
            }
        
        try:
            # Read original image
            original_image = cv2.imread(image_path)
            if original_image is None:
                return {
                    "error": "Cannot read image file",
                    "detection_count": 0,
                    "bboxes": [],
                    "confidences": [],
                    "track_ids": [],
                    "detections": [],
                    "annotated_image_path": image_path,
                    "average_confidence": 0.0
                }
            
            # Use tracking mode
            results = self.model.track(original_image, persist=True, tracker="bytetrack.yaml")
            result = results[0]
            
            detections = []
            bboxes = []
            confidences = []
            track_ids = []
            
            if result.boxes is not None and result.boxes.id is not None:
                # Case with tracking IDs
                for box, track_id in zip(result.boxes, result.boxes.id):
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    confidence = box.conf[0].cpu().numpy().item()
                    class_id = int(box.cls[0].cpu().numpy())
                    track_id = int(track_id.cpu().numpy())
                    
                    detection = {
                        "bbox": [round(coord, 2) for coord in bbox],
                        "confidence": round(confidence, 4),
                        "class_id": class_id,
                        "track_id": track_id
                    }
                    detections.append(detection)
                    bboxes.append(bbox)
                    confidences.append(confidence)
                    track_ids.append(track_id)
                    
                    # Update tracking history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # Draw tracking path
                    self.draw_tracking_path(original_image, bbox, track_id)
                    
                    # Draw detection box with confidence (with tracking ID)
                    self.draw_detection_box(original_image, bbox, confidence, track_id)
            
            # Save annotated image
            annotated_filename = f"annotated_{os.path.basename(image_path)}"
            annotated_path = f"uploaded_images/{annotated_filename}"
        
            # 确保目录存在
            os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        
            # 保存图片
            success = cv2.imwrite(annotated_path, original_image)
            if not success:
                print(f"Warning: Failed to save annotated image to {annotated_path}")
                annotated_path = image_path
        
            return {
                "detection_count": len(detections),
                "bboxes": bboxes,
                "confidences": confidences,
                "track_ids": track_ids,
                "detections": detections,
                "annotated_image_path": annotated_path,
                "annotated_image_url": f"/uploaded_images/{annotated_filename}",
                "average_confidence": round(np.mean(confidences).item(), 4) if confidences else 0.0
            }
        
        except Exception as e:
            print(f"Detection error: {e}")
            return {
                "error": f"Detection failed: {str(e)}",
                "detection_count": 0,
                "bboxes": [],
                "confidences": [],
                "track_ids": [],
                "detections": [],
                "annotated_image_path": image_path,
                "average_confidence": 0.0
            }

    def detect_video(self, video_path: str, sample_interval: int = 5) -> Dict:
        """Video detection using tracking"""
        # 播放检测开始音频
        self.play_detection_sound()
        
        self.load_model()
        
        if self.model is None:
            return {
                "error": f"Model not available: {self.model_error}",
                "total_frames": 0,
                "analyzed_frames": 0,
                "total_detections": 0,
                "average_detections_per_frame": 0,
                "video_duration": 0,
                "fps": 0,
                "sample_interval": sample_interval,
                "unique_tracks": 0,
                "frames_summary": [],
                "annotated_video_path": video_path
            }
        
        # 原有的视频检测逻辑保持不变...
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_summary = []
        total_detections = 0
        frame_count = 0
        
        output_video_path = f"uploaded_videos/annotated_{os.path.basename(video_path)}"
        os.makedirs("uploaded_videos", exist_ok=True)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        self.track_history = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame_detections = 0
            frame_confidences = []
            frame_track_ids = []
            
            if frame_count % sample_interval == 0:
                results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
                result = results[0]
                
                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes, result.boxes.id):
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        confidence = box.conf[0].cpu().numpy().item()
                        track_id = int(track_id.cpu().numpy())
                        
                        if confidence < 0.689:
                            continue
                            
                        frame_confidences.append(confidence)
                        frame_track_ids.append(track_id)
                        
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        
                        self.draw_tracking_path(frame, bbox, track_id)
                        self.draw_detection_box(frame, bbox, confidence, track_id)
                        
                        frame_detections += 1
                
                frames_summary.append({
                    "frame_number": frame_count,
                    "timestamp": round(frame_count / fps, 2),
                    "detection_count": frame_detections,
                    "track_ids": frame_track_ids,
                    "average_confidence": round(np.mean(frame_confidences).item(), 4) if frame_confidences else 0.0
                })
                
                total_detections += frame_detections
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        analyzed_frames = len(frames_summary)
        avg_detections = round(total_detections / analyzed_frames, 2) if analyzed_frames > 0 else 0
        
        return {
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "total_detections": total_detections,
            "average_detections_per_frame": avg_detections,
            "video_duration": round(total_frames / fps, 2),
            "fps": fps,
            "sample_interval": sample_interval,
            "unique_tracks": len(set([id for frame in frames_summary for id in frame.get('track_ids', [])])),
            "frames_summary": frames_summary,
            "annotated_video_path": output_video_path
        }

    # 其他辅助方法保持不变...
    def draw_tracking_path(self, image, bbox, track_id):
        """Draw tracking trajectory"""
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        self.track_history[track_id].append((center_x, center_y))
        
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)
        
        points = np.hstack(self.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=False, color=(230, 230, 230), thickness=2)
    
    def draw_detection_box(self, image, bbox, confidence, track_id):
        """Draw detection box and label"""
        x1, y1, x2, y2 = map(int, bbox)
        
        color = self.get_track_color(track_id)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{track_id} Aluminum:{confidence:.3f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def get_track_color(self, track_id):
        """Generate color based on tracking ID"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
        ]
        return colors[track_id % len(colors)]

detector = AluminumDustDetector()