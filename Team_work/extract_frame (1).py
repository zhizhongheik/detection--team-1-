import cv2
import os
import argparse
import csv
from datetime import timedelta


def extract_frames(video_path, output_dir, frame_interval=1, skip_seconds=0, gray=False, csv_file=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = int(skip_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    count = skip_frames
    saved = 0
    os.makedirs(output_dir, exist_ok=True)

    # Creat csv file
    csv_path = None
    csv_writer = None
    if csv_file:
        csv_path = os.path.join(output_dir, csv_file)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame Number', 'Timestamp', 'File Path'])
    
    frames_to_show = []

    while True:
        ret, frame = cap.read()
        if not ret or count >= total_frames:
            break


        if count % frame_interval == 0:
            if gray:
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_with_watermark = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)
            else:
                frame_processed = frame.copy()
                frame_with_watermark = frame_processed.copy()
            
            
            timestamp_seconds = count / fps
            timestamp_str = str(timedelta(seconds=int(timestamp_seconds)))
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_with_watermark, f"Time: {timestamp_str}", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_watermark, f"Frame: {count}", (10, 60), font, 0.7, (0, 255, 0), 2)
            
            filename = f"frame_{saved:03}.jpg"
            path = os.path.join(output_dir, filename)
            
            
            cv2.imwrite(path, frame_with_watermark)
                
            print(f"Saved: {path}")
            
            # save 3 frame count
            if saved < 3:
                frames_to_show.append((frame_with_watermark, path))
            
            
            if csv_file:
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([count, timestamp_str, path])
            
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames.")
    
    for i, (frame, path) in enumerate(frames_to_show):
        cv2.imshow(f'Extracted Frame {i+1}: {os.path.basename(path)}', frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="frames", help="Output folder")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval for extraction (extract every N frames)")
    parser.add_argument("--skip_seconds", type=float, default=0, help="Skip the first N seconds of the video")
    parser.add_argument("--gray", action="store_true", help="Extract frames in grayscale")
    parser.add_argument("--csv", default="frames.csv", help="CSV file to store frame paths")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.out, args.frame_interval, args.skip_seconds, args.gray, args.csv)