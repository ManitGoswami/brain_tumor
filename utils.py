import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import BrainTumorClassifier  # Ensure you have the same model structure as in your main.py
from PIL import Image
import numpy as np

def load_model():
    # Ensure the correct device is set for loading the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model
    model = BrainTumorClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load("brain_tumor_classifier.pth", map_location=device))
    model.eval()

    # Define the transformation (as you were using for evaluation)
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Class names (adjust if needed to match your dataset)
    class_names = ["class1", "class2", "class3", "class4"]  # Replace with your actual class names
    
    return model, transform, class_names

def predict(model, image, transform, class_names):
    # Apply the transformation
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    return class_names[predicted_class.item()]
