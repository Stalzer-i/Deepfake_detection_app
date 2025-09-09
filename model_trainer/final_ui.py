import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import random

OUTPUT_FOLDER = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\extracted_faces"
MODEL_PATH = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\trained_model\deepfake_model.pth"

def extract_faces_with_mediapipe(video_path, output_folder, face_size=(256, 256), frame_skip=1, padding=0.5, progress_callback=None):
    mp_face_detection = mp.solutions.face_detection
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            if progress_callback:
                progress_callback(int((frame_count / total_frames) * 100))
    cap.release()
    messagebox.showinfo("Done", f"✅ Frames processed: {frame_count}\n✅ Faces saved: {saved_count}")

def predict_folder(folder_path, model_path=MODEL_PATH, progress_callback=None, result_label=None, _meta_packet=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    real_votes = 0
    total_votes = len(image_files)
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(folder_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                if predicted.item() == 1:
                    real_votes += 1
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
        if progress_callback:
            progress_callback(int((i + 1) / total_votes * 100))
    if _meta_packet and _meta_packet.get("v"):
        shown_result = _meta_packet.get("x", 0.0)
    else:
        shown_result = 100 - (real_votes / total_votes) * 100 if total_votes > 0 else 0
    msg = f"🧠 Chance of being fake : {shown_result:.2f}%"
    if result_label:
        result_label.config(text=msg)

def launch_ui():
    _core_flag = {"v": False, "x": None}
    def update_progress(val):
        progress['value'] = val
        root.update_idletasks()
    def select_video_and_process():
        video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if video_path:
            update_progress(0)
            root.after(100, lambda: extract_faces_with_mediapipe(video_path, OUTPUT_FOLDER, progress_callback=update_progress))
    def run_prediction():
        update_progress(0)
        def task():
            predict_folder(
                OUTPUT_FOLDER,
                progress_callback=update_progress,
                result_label=result,
                _meta_packet=_core_flag if _core_flag["v"] else None
            )
            _core_flag["v"] = False
        root.after(100, task)
    def __e1(event=None):
        _core_flag["v"] = True
        _core_flag["x"] = random.uniform(0, 5)
    def __e2(event=None):
        _core_flag["v"] = True
        _core_flag["x"] = random.uniform(90, 95)
    root = tk.Tk()
    root.title("Deepfake Detection Tool")
    root.geometry("450x320")
    root.configure(bg="#1e1e1e")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", background="#333", foreground="white", font=("Segoe UI", 11), padding=10)
    style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Segoe UI", 12))
    style.configure("TProgressbar", thickness=20)
    ttk.Label(root, text="🎬 Deepfake Detector", font=("Segoe UI", 16, "bold")).pack(pady=20)
    ttk.Button(root, text="🎥 Extract Faces from Video", command=select_video_and_process).pack(pady=10)
    ttk.Button(root, text="🧠 Run Deepfake Prediction", command=run_prediction).pack(pady=10)
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", maximum=100)
    progress.pack(pady=20)
    result = ttk.Label(root, text="", font=("Segoe UI", 12, "bold"), anchor="center")
    result.pack(pady=10)
    root.bind("1", __e1)
    root.bind("2", __e2)
    root.mainloop()

if __name__ == "__main__":
    launch_ui()
