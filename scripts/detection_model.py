from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import List, Dict, Any

class AluminumDustDetector:
    def __init__(self, model_path: str = "/app/scripts/model/best.pt"):
        self.model_path = model_path
        self.model = None  # 延迟加载
        self.track_history = {}
        self.model_loaded = False
        self.model_error = None
    
    def load_model(self):
        """延迟加载模型"""
        if self.model is None and not self.model_loaded:
            try:
                print(f"Loading model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.model_loaded = True
                print("Model loaded successfully!")
            except Exception as e:
                self.model_error = str(e)
                print(f"Error loading model: {e}")
                self.model = None
                self.model_loaded = False
    
    def detect(self, image_path: str) -> Dict:
        """Detection method using tracking"""
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
        
        # Read original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise Exception("Cannot read image file")
        
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
        annotated_path = f"/app/uploaded_images/{annotated_filename}"  # 使用绝对路径
    
        # 确保目录存在
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
    
        # 保存图片
        success = cv2.imwrite(annotated_path, original_image)
        if not success:
          print(f"Warning: Failed to save annotated image to {annotated_path}")
           # 如果保存失败，使用原路径
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
    
    def draw_tracking_path(self, image, bbox, track_id):
        """Draw tracking trajectory"""
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        # Add current center point to history
        self.track_history[track_id].append((center_x, center_y))
        
        # Keep only recent 30 points
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)
        
        # Draw trajectory line
        points = np.hstack(self.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=False, color=(230, 230, 230), thickness=2)
    
    def draw_detection_box(self, image, bbox, confidence, track_id):
        """Draw detection box and label"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Select color based on tracking ID
        color = self.get_track_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Build label text
        label = f"ID:{track_id} Aluminum:{confidence:.3f}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def get_track_color(self, track_id):
        """Generate color based on tracking ID"""
        colors = [
            (0, 255, 0),    # Green - 主要
            (255, 0, 0),    # Blue - 备用
        ]
        return colors[track_id % len(colors)]
    
    def detect_video(self, video_path: str, sample_interval: int = 5) -> Dict:
        """Video detection using tracking"""
        self.load_model()  # 确保模型已加载
        
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
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_summary = []
        total_detections = 0
        frame_count = 0
        
        # Video output settings
        output_video_path = f"uploaded_videos/annotated_{os.path.basename(video_path)}"
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Reset tracking history
        self.track_history = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            frame_detections = 0
            frame_confidences = []
            frame_track_ids = []
            
            # Detect at sampling interval
            if frame_count % sample_interval == 0:
                # Use tracking detection
                results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
                result = results[0]
                
                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id in zip(result.boxes, result.boxes.id):
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        confidence = box.conf[0].cpu().numpy().item()
                        track_id = int(track_id.cpu().numpy())
                        
                        # Apply confidence threshold
                        if confidence < 0.689:  # Adjustable
                            continue
                            
                        frame_confidences.append(confidence)
                        frame_track_ids.append(track_id)
                        
                        # Update tracking history
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        
                        # Draw tracking path
                        self.draw_tracking_path(frame, bbox, track_id)
                        
                        # Draw detection box
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
            
            # Write processed frame
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Calculate statistics
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

# 修改这里，不立即初始化模型
detector = AluminumDustDetector()