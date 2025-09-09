import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

def load_model(model_path, device):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Assuming two classes: Real and Fake
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    classes = ["Real", "Fake"]
    print(f"Prediction: {classes[predicted.item()]}")

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\models\15e_0.0001lr_res18\deepfake_model.pth"  # Change this to your model path
    IMAGE_PATH = r"C:\Users\sanjo\OneDrive\Desktop\Project_2\to_be__predicted\test.jpg"# Change this to your image path
    
    predict(IMAGE_PATH, MODEL_PATH)
