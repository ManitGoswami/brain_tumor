import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torchvision import transforms
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# FastAPI app
app = FastAPI()

# CORS settings to allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom activation
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# CBAM block
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            Mish(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        c = self.channel(x)
        x = x * c
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa

# DropBlock
class DropBlock(nn.Module):
    def __init__(self, p=0.1):
        super(DropBlock, self).__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        return self.drop(x)

# Brain Tumor Classifier Model
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(4, out_ch),
                Mish(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(4, out_ch),
                Mish(),
                CBAM(out_ch),
                nn.MaxPool2d(2),
                DropBlock(0.1)
            )

        self.backbone = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            Mish(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# Load the model
model = BrainTumorClassifier(num_classes=4)
model.load_state_dict(torch.load("brain_tumor_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocessing
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("L")
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        tensor_image = preprocess_image(image_bytes)
        with torch.no_grad():
            output = model(tensor_image)
            _, predicted_class = torch.max(output, 1)
            class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
            predicted_label = class_names[predicted_class.item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() * 100
            return {
                'success': True,
                'predicted_result': predicted_label,
                'confidence': f'{confidence:.2f}%',
                'message': f'Status of the Brain Image: {predicted_label} with a confidence of {confidence:.2f}%',
                'raw_confidence': confidence,
                'class_index': predicted_class.item()
            }
    except Exception as e:
        return {
            'success': False,
            'predicted_result': None,
            'confidence': None,
            'message': f'Prediction failed: {str(e)}'
        }
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root "/"
@app.get("/")
def read_root():
    return FileResponse("static/index.html")
