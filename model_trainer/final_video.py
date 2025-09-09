import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision import models
from PIL import Image
from tqdm import tqdm  
import random

MODEL_PATH = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\trained_model\deepfake_model.pth"

def predict_folder(folder_path, model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No valid image files found in the folder.")
        return

    real_count = 0
    total_count = 0

    for img_name in tqdm(image_files, desc="Predicting images"):
        img_path = os.path.join(folder_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                if predicted.item() == 0:  # assuming class 0 is 'real'
                    real_count += 1
            total_count += 1
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    real_percentage = 100 - (real_count / total_count) * 100 if total_count > 0 else 0
    print(f"Real images: {real_percentage:.2f}% ({real_count}/{total_count})")


if __name__ == "__main__":
    predict_folder(r"C:\Users\sanjo\OneDrive\Desktop\Project_2\extracted_faces")
 
