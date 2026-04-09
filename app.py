from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import io
import base64
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from model import UNet

load_dotenv()

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "house_segmentation_model.pth")
PORT = int(os.getenv("PORT", 5000))
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")

app.config['SECRET_KEY'] = SECRET_KEY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

model = None

def load_model():
    global model
    if model is None:
        model = UNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    return model

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "House segmentation API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    
    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        
        current_model = load_model()

        with torch.no_grad():
            output = current_model(image_tensor)
            probs = torch.sigmoid(output)
            pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()        

        pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

        buffer = io.BytesIO()
        pred_mask_img.save(buffer, format="PNG")
        buffer.seek(0)

        encoded_mask = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({
            "message": "Segmentation completed",
            "predicted_mask": encoded_mask
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)