from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import io

load_dotenv()

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "house_segmentation_model.pth")
PORT = int(os.getenv("PORT", 5000))
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")

app.config['SECRET_KEY'] = SECRET_KEY

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)

        return self.final(u2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400
    
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float().squeeze().cpu().numpy()

    building_pixels = int(pred_mask.sum())
    total_pixels = int(pred_mask.size)
    building_ratio = building_pixels / total_pixels

    return jsonify({
        "building_pixels": building_pixels,
        "total_pixels": total_pixels,
        "building_ratio": building_ratio
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)