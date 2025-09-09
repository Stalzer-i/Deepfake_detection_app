import cv2
import mediapipe as mp
import os
from PIL import Image

def extract_faces_with_mediapipe(video_path, output_folder, face_size=(256, 256), frame_skip=1, padding=0.5):
    mp_face_detection = mp.solutions.face_detection

    os.makedirs(output_folder, exist_ok=True)
    
    # Delete all images in the folder before processing
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                height, width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)

                        pad_w = int(w * padding)
                        pad_h = int(h * padding)
                        x1 = max(x - pad_w, 0)
                        y1 = max(y - pad_h, 0)
                        x2 = min(x + w + pad_w, width)
                        y2 = min(y + h + pad_h, height)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        face = frame[y1:y2, x1:x2]
                        if face.shape[0] < 50 or face.shape[1] < 50:
                            continue

                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        face_pil = face_pil.resize(face_size)
                        save_path = os.path.join(output_folder, f"face_{frame_count}_{i}.jpg")

                        try:
                            face_pil.save(save_path)
                            saved_count += 1
                        except Exception as e:
                            print(f"Failed to save {save_path}: {e}")

            frame_count += 1

    cap.release()
    print(f"\n✅ Done. Processed frames: {frame_count}")
    print(f"✅ Faces successfully saved: {saved_count}")

if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\sanjo\OneDrive\Desktop\videoplayback.mp4"
    OUTPUT_FOLDER = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\extracted_faces"

    extract_faces_with_mediapipe(VIDEO_PATH, OUTPUT_FOLDER)
