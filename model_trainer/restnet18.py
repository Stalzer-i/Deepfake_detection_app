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

DATA_DIR = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\dataset" 
MODEL_PATH = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\trained_model\deepfake_model.pth"


def load_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.class_to_idx, val_dataset


def train_model(data_dir, epochs=5, batch_size=32, model_path=MODEL_PATH):
    train_loader, val_loader, class_to_idx, _ = load_data(data_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # two classes : real and fake
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation loss calculation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


def predict(model_path=MODEL_PATH, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    _, _, class_to_idx, val_dataset = load_data(DATA_DIR)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Select 100 random images from dataset
    sampled_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    sampled_subset = Subset(val_dataset, sampled_indices)
    sample_loader = DataLoader(sampled_subset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for image, label in sample_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            
            if predicted.item() == label.item():
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100
    print(f"Accuracy on {total} random images: {accuracy:.2f}%")


if __name__ == "__main__":
    train_model(DATA_DIR, epochs=5)
    predict()