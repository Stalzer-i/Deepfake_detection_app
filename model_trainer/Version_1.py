# deepfake_tk_app.py
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision import models
import random
import json
import shutil
import tempfile
from tqdm import tqdm

# --- Mediapipe & cv imports for video face extraction ---
import cv2
import mediapipe as mp

# ---------------------------
# Utilities: model factory
# ---------------------------
def get_model_by_name(name, num_classes=2, pretrained=True):
    """Return a torch model (uninitialized classifier head) for a given architecture name."""
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == "resnet34":
        m = models.resnet34(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == "resnet50":
        m = models.resnet50(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == "resnext50_32x4d":
        m = models.resnext50_32x4d(pretrained=pretrained)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(pretrained=pretrained)
        in_f = m.classifier[-1].in_features
        # replace classifier
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name == "densenet121":
        m = models.densenet121(pretrained=pretrained)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m

# ---------------------------
# Data loader / transforms
# ---------------------------
IMG_SIZE = (224, 224)
DEFAULT_BATCH = 32

def make_data_loaders(data_dir, batch_size=DEFAULT_BATCH):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    if len(dataset) < 2:
        raise ValueError("Dataset folder seems too small or not organized as ImageFolder (class subfolders).")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset.class_to_idx, val_dataset

# ---------------------------
# Save / Load model helpers
# ---------------------------
def save_model(model, arch_name, path):
    payload = {'arch': arch_name, 'state_dict': model.state_dict()}
    torch.save(payload, path)

def load_model_from_file(model_path, device):
    """
    Loads model. We expect either:
      - a dict {'arch': <name>, 'state_dict': ...} saved by this app, OR
      - a plain state_dict (fallback; user must select arch manually)
    Returns (model, arch_name or None)
    """
    checkpoint = torch.load(model_path, map_location=device)
    arch = None
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and 'arch' in checkpoint:
        arch = checkpoint['arch']
        state = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and any(k.startswith('layer') or k.startswith('conv') or k.startswith('fc') for k in checkpoint.keys()):
        # likely a raw state_dict
        state = checkpoint
    else:
        # unknown format
        state = checkpoint

    if arch is not None:
        model = get_model_by_name(arch, num_classes=2, pretrained=False)
    else:
        # arch unknown; caller should handle by building model externally
        return None, None, state

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, arch, state

# ---------------------------
# Training routine
# ---------------------------
def train_worker(data_dir, arch_name, epochs, batch_size, lr, save_path, log_callback, stop_event):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_callback(f"Using device: {device}")
        train_loader, val_loader, class_to_idx, _ = make_data_loaders(data_dir, batch_size)
        model = get_model_by_name(arch_name, num_classes=2, pretrained=True)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            if stop_event.is_set():
                log_callback("Training stopped by user.")
                return
            model.train()
            running_loss = 0.0
            n_batches = 0
            log_callback(f"Starting epoch {epoch+1}/{epochs}")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            avg_train_loss = running_loss / max(1, n_batches)

            # validation
            model.eval()
            val_loss = 0.0
            n_val = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    n_val += 1
            avg_val_loss = val_loss / max(1, n_val)
            acc = (correct / total) * 100 if total>0 else 0.0
            log_callback(f"Epoch {epoch+1}/{epochs}  TrainLoss: {avg_train_loss:.4f}  ValLoss: {avg_val_loss:.4f}  ValAcc: {acc:.2f}%")

        # save model
        save_model(model, arch_name, save_path)
        log_callback(f"Model saved to: {save_path}")
    except Exception as e:
        log_callback(f"Training error: {e}")

# ---------------------------
# Prediction helpers
# ---------------------------
from torch.nn.functional import softmax

def predict_image_with_model(image_path, model_obj, device):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model_obj(tensor)
        probs = softmax(out[0], dim=0).cpu().numpy()
        pred_idx = int(torch.argmax(out, 1).item())
    # order: we assume [Real, Fake] but dataset class order depends on training dataset; user should ensure
    return pred_idx, probs

# ---------------------------
# Mediapipe face extraction (video -> faces)
# ---------------------------
def extract_faces_with_mediapipe(video_path, output_folder, face_size=(224,224), frame_skip=1, padding=0.5, log_callback=None):
    mp_face_detection = mp.solutions.face_detection
    os.makedirs(output_folder, exist_ok=True)
    # clear output folder
    for filename in os.listdir(output_folder):
        fp = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception:
            pass
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
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb)
                if results.detections:
                    for i, det in enumerate(results.detections):
                        bbox = det.location_data.relative_bounding_box
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
                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize(face_size)
                        save_path = os.path.join(output_folder, f"face_{frame_count}_{i}.jpg")
                        try:
                            face_pil.save(save_path)
                            saved_count += 1
                        except Exception as e:
                            if log_callback:
                                log_callback(f"Failed to save {save_path}: {e}")
            frame_count += 1
    cap.release()
    if log_callback:
        log_callback(f"Processed frames: {frame_count}, faces saved: {saved_count}")
    return saved_count

# ---------------------------
# Tkinter UI
# ---------------------------
class DeepfakeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Trainer & Predictor")
        self.geometry("980x720")
        # state
        self.stop_event = threading.Event()

        self.available_models = ["resnet18", "resnet34", "resnet50", "resnext50_32x4d", "mobilenet_v2", "densenet121"]

        # --- Setup UI frames ---
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # Train tab
        train_frame = ttk.Frame(notebook)
        self.build_train_tab(train_frame)
        notebook.add(train_frame, text="Train Model")

        # Image predict tab
        img_frame = ttk.Frame(notebook)
        self.build_image_tab(img_frame)
        notebook.add(img_frame, text="Predict Image")

        # Video predict tab
        vid_frame = ttk.Frame(notebook)
        self.build_video_tab(vid_frame)
        notebook.add(vid_frame, text="Predict Video (extract faces)")

        # Log panel
        log_frame = ttk.LabelFrame(self, text="Log / Output")
        log_frame.pack(fill="both", expand=False, padx=8, pady=(0,8))
        self.log_text = tk.Text(log_frame, height=12)
        self.log_text.pack(fill="both", expand=True)
        self.log("App started. Select a tab to begin.")

    def log(self, text):
        ts = time.strftime("%H:%M:%S")
        def _append():
            self.log_text.insert("end", f"[{ts}] {text}\n")
            self.log_text.see("end")
        # always schedule on main thread
        self.after(0, _append)

    # -----------------
    # Train tab UI
    # -----------------
    def build_train_tab(self, parent):
        pad = {'padx':6, 'pady':6}
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Dataset folder (ImageFolder format):").grid(row=0, column=0, sticky="w", **pad)
        self.train_data_entry = ttk.Entry(top, width=60)
        self.train_data_entry.grid(row=0, column=1, **pad)
        ttk.Button(top, text="Browse", command=self.browse_train_folder).grid(row=0, column=2, **pad)

        ttk.Label(top, text="Choose model architecture:").grid(row=1, column=0, sticky="w", **pad)
        self.arch_var = tk.StringVar(value=self.available_models[1])
        arch_dropdown = ttk.OptionMenu(top, self.arch_var, self.available_models[1], *self.available_models)
        arch_dropdown.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(top, text="Epochs:").grid(row=2, column=0, sticky="w", **pad)
        self.epochs_var = tk.IntVar(value=5)
        ttk.Entry(top, textvariable=self.epochs_var, width=10).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(top, text="Batch size:").grid(row=3, column=0, sticky="w", **pad)
        self.batch_var = tk.IntVar(value=32)
        ttk.Entry(top, textvariable=self.batch_var, width=10).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(top, text="Learning rate:").grid(row=4, column=0, sticky="w", **pad)
        self.lr_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(top, textvariable=self.lr_var, width=12).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(top, text="Save model to:").grid(row=5, column=0, sticky="w", **pad)
        self.save_model_entry = ttk.Entry(top, width=60)
        self.save_model_entry.grid(row=5, column=1, **pad)
        ttk.Button(top, text="Browse", command=self.browse_save_model).grid(row=5, column=2, **pad)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=8, pady=6)
        self.train_button = ttk.Button(btn_frame, text="Start Training", command=self.start_training_thread)
        self.train_button.pack(side="left", padx=6)
        self.stop_train_button = ttk.Button(btn_frame, text="Stop Training", command=self.stop_training)
        self.stop_train_button.pack(side="left", padx=6)

    def browse_train_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.train_data_entry.delete(0, 'end')
            self.train_data_entry.insert(0, d)

    def browse_save_model(self):
    # Let user pick a folder (location) and create a sensible default filename there
        d = filedialog.askdirectory()
        if d:
        # create a default filename using architecture + timestamp so it won't overwrite
            arch = self.arch_var.get() if hasattr(self, 'arch_var') else "model"
            filename = f"deepfake_{arch}_{int(time.time())}.pth"
            fullpath = os.path.join(d, filename)
            self.save_model_entry.delete(0, 'end')
            self.save_model_entry.insert(0, fullpath)


    def start_training_thread(self):
        data_dir = self.train_data_entry.get().strip()
        save_path = self.save_model_entry.get().strip()
        arch = self.arch_var.get()
        epochs = int(self.epochs_var.get())
        batch_size = int(self.batch_var.get())
        lr = float(self.lr_var.get())

        if not os.path.isdir(data_dir):
            messagebox.showerror("Dataset error", "Please choose a valid dataset folder (ImageFolder style: class subfolders).")
            return
        if not save_path:
            messagebox.showerror("Save path", "Please choose a path to save the trained model.")
            return

        self.stop_event.clear()
        t = threading.Thread(target=train_worker, args=(data_dir, arch, epochs, batch_size, lr, save_path, self.log, self.stop_event), daemon=True)
        t.start()
        self.log(f"Training started (arch={arch}, epochs={epochs}, batch={batch_size}, lr={lr}).")

    def stop_training(self):
        self.stop_event.set()
        self.log("Stop signal sent to training.")

    # -----------------
    # Image predict tab
    # -----------------
    def build_image_tab(self, parent):
        pad = {'padx':6, 'pady':6}
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=8, pady=8)

        ttk.Label(frame, text="Choose image:").grid(row=0, column=0, sticky="w", **pad)
        self.image_entry = ttk.Entry(frame, width=60)
        self.image_entry.grid(row=0, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self.browse_image).grid(row=0, column=2, **pad)

        ttk.Label(frame, text="Choose trained model (.pth):").grid(row=1, column=0, sticky="w", **pad)
        self.model_entry = ttk.Entry(frame, width=60)
        self.model_entry.grid(row=1, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self.browse_model_file).grid(row=1, column=2, **pad)

        ttk.Label(frame, text="If model file lacks arch, choose architecture:").grid(row=2, column=0, sticky="w", **pad)
        self.img_arch_var = tk.StringVar(value=self.available_models[1])
        arch_dropdown = ttk.OptionMenu(frame, self.img_arch_var, self.available_models[1], *self.available_models)
        arch_dropdown.grid(row=2, column=1, sticky="w", **pad)

        btn = ttk.Button(parent, text="Run Image Prediction", command=self.run_image_prediction_thread)
        btn.pack(padx=8, pady=6)

        # image preview canvas
        self.preview_label = ttk.Label(parent)
        self.preview_label.pack(padx=8, pady=6)

    def browse_image(self):
        f = filedialog.askopenfilename(filetypes=[("Images","*.jpg;*.jpeg;*.png;*.bmp"),("All files","*.*")])
        if f:
            self.image_entry.delete(0, 'end')
            self.image_entry.insert(0, f)
            try:
                img = Image.open(f).resize((200,200))
                imgtk = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=imgtk)
                self.preview_label.image = imgtk
            except Exception:
                pass

    def browse_model_file(self):
        f = filedialog.askopenfilename(filetypes=[("PyTorch models","*.pth;*.pt"),("All files","*.*")])
        if f:
            self.model_entry.delete(0, 'end')
            self.model_entry.insert(0, f)

    def run_image_prediction_thread(self):
        image_path = self.image_entry.get().strip()
        model_path = self.model_entry.get().strip()
        fallback_arch = self.img_arch_var.get()
        if not os.path.isfile(image_path):
            messagebox.showerror("Image", "Choose a valid image file.")
            return
        if not os.path.isfile(model_path):
            messagebox.showerror("Model", "Choose a valid model file.")
            return
        t = threading.Thread(target=self.image_prediction_worker, args=(image_path, model_path, fallback_arch), daemon=True)
        t.start()

    def image_prediction_worker(self, image_path, model_path, fallback_arch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"Loading model from {model_path} on {device} ...")
        loaded = torch.load(model_path, map_location=device)
        model = None
        arch = None
        # try to interpret saved object
        if isinstance(loaded, dict) and 'arch' in loaded and 'state_dict' in loaded:
            arch = loaded['arch']
            state = loaded['state_dict']
            model = get_model_by_name(arch, num_classes=2, pretrained=False)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
        else:
            # fallback: raw state_dict -> use fallback arch selected by user
            state = loaded
            arch = fallback_arch
            model = get_model_by_name(arch, num_classes=2, pretrained=False)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()

        pred_idx, probs = predict_image_with_model(image_path, model, device)
        labels = ["Real", "Fake"]
        result_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        self.log(f"Image prediction: {result_label}  probs={probs}")
        messagebox.showinfo("Image Prediction", f"Prediction: {result_label}\nProbabilities: {probs}")

    # -----------------
    # Video predict tab
    # -----------------
    def build_video_tab(self, parent):
        pad = {'padx':6, 'pady':6}
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=8, pady=8)

        ttk.Label(frame, text="Choose video file:").grid(row=0, column=0, sticky="w", **pad)
        self.video_entry = ttk.Entry(frame, width=60)
        self.video_entry.grid(row=0, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self.browse_video).grid(row=0, column=2, **pad)

        ttk.Label(frame, text="Choose trained model (.pth):").grid(row=1, column=0, sticky="w", **pad)
        self.video_model_entry = ttk.Entry(frame, width=60)
        self.video_model_entry.grid(row=1, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self.browse_video_model).grid(row=1, column=2, **pad)

        ttk.Label(frame, text="If model file lacks arch, choose architecture:").grid(row=2, column=0, sticky="w", **pad)
        self.vid_arch_var = tk.StringVar(value=self.available_models[1])
        arch_dropdown = ttk.OptionMenu(frame, self.vid_arch_var, self.available_models[1], *self.available_models)
        arch_dropdown.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(frame, text="Frame skip (1 = every frame, 2 = every 2nd, ...):").grid(row=3, column=0, sticky="w", **pad)
        self.frame_skip_var = tk.IntVar(value=2)
        ttk.Entry(frame, textvariable=self.frame_skip_var, width=8).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(frame, text="Padding around face (0-1):").grid(row=4, column=0, sticky="w", **pad)
        self.padding_var = tk.DoubleVar(value=0.5)
        ttk.Entry(frame, textvariable=self.padding_var, width=8).grid(row=4, column=1, sticky="w", **pad)

        btn = ttk.Button(parent, text="Extract Faces & Predict Video", command=self.run_video_prediction_thread)
        btn.pack(padx=8, pady=6)

    def browse_video(self):
        f = filedialog.askopenfilename(filetypes=[("Videos","*.mp4;*.mov;*.avi;*.mkv"),("All files","*.*")])
        if f:
            self.video_entry.delete(0, 'end')
            self.video_entry.insert(0, f)

    def browse_video_model(self):
        f = filedialog.askopenfilename(filetypes=[("PyTorch models","*.pth;*.pt"),("All files","*.*")])
        if f:
            self.video_model_entry.delete(0, 'end')
            self.video_model_entry.insert(0, f)

    def run_video_prediction_thread(self):
        video_path = self.video_entry.get().strip()
        model_path = self.video_model_entry.get().strip()
        if not os.path.isfile(video_path):
            messagebox.showerror("Video", "Choose a valid video file.")
            return
        if not os.path.isfile(model_path):
            messagebox.showerror("Model", "Choose a valid model file.")
            return
        frame_skip = int(self.frame_skip_var.get())
        padding = float(self.padding_var.get())
        fallback_arch = self.vid_arch_var.get()
        t = threading.Thread(target=self.video_prediction_worker, args=(video_path, model_path, fallback_arch, frame_skip, padding), daemon=True)
        t.start()

    def video_prediction_worker(self, video_path, model_path, fallback_arch, frame_skip, padding):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"Loading model from {model_path} ...")
        loaded = torch.load(model_path, map_location=device)
        model = None
        arch = None
        if isinstance(loaded, dict) and 'arch' in loaded and 'state_dict' in loaded:
            arch = loaded['arch']
            state = loaded['state_dict']
            model = get_model_by_name(arch, num_classes=2, pretrained=False)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
        else:
            state = loaded
            arch = fallback_arch
            model = get_model_by_name(arch, num_classes=2, pretrained=False)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()

        # extract faces to temp dir
        tmpdir = tempfile.mkdtemp(prefix="df_faces_")
        self.log(f"Extracting faces to {tmpdir} ...")
        count = extract_faces_with_mediapipe(video_path, tmpdir, face_size=(224,224), frame_skip=frame_skip, padding=padding, log_callback=self.log)
        if count == 0:
            self.log("No faces extracted. Aborting video prediction.")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return

        # predict all faces
        self.log("Predicting extracted faces ...")
        fake_count = 0
        real_count = 0
        probs_list = []
        for fn in os.listdir(tmpdir):
            if not fn.lower().endswith((".jpg",".png",".jpeg")):
                continue
            fp = os.path.join(tmpdir, fn)
            try:
                pred_idx, probs = predict_image_with_model(fp, model, device)
                probs_list.append(probs)
                if pred_idx == 0:
                    # careful: depends on how labels were ordered during training. We assume 0 -> Real, 1 -> Fake.
                    real_count += 1
                else:
                    fake_count += 1
            except Exception as e:
                self.log(f"Failed to predict {fp}: {e}")

        total = real_count + fake_count
        fake_pct = (fake_count / total) * 100 if total > 0 else 0.0
        real_pct = (real_count / total) * 100 if total > 0 else 0.0
        self.log(f"Video result: total_faces={total}, real={real_count}, fake={fake_count}, fake%={fake_pct:.2f}%, real%={real_pct:.2f}%")

        messagebox.showinfo("Video Prediction", f"Faces analyzed: {total}\nReal: {real_count}\nFake: {fake_count}\nFake%: {fake_pct:.2f}%")

        shutil.rmtree(tmpdir, ignore_errors=True)



if __name__ == "__main__":
    app = DeepfakeApp()
    app.mainloop()
