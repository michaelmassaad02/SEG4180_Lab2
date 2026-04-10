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

# Load environment variables from .env file (for secrets/config)
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Load configuration values from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "house_segmentation_model.pth")
PORT = int(os.getenv("PORT", 5000))
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")

# Set Flask secret key (used for session security)
app.config['SECRET_KEY'] = SECRET_KEY

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing transformations
# Resize input image and convert to tensor
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Global model variable (lazy loading)
model = None

def load_model():
    """
    Loads the trained UNet model into memory only once.
    Uses lazy loading to avoid reloading the model for every request.
    """
    global model
    if model is None:
        model = UNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    return model

@app.route("/", methods=["GET"])
def home():
    """
    Health check endpoint.
    Confirms that the API is running.
    """
    return jsonify({"message": "House segmentation API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an image file via POST request and returns a segmentation mask.
    The mask is returned as a base64-encoded PNG image.
    """
    # Check if image is provided in the request
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    
    try:
        # Load and preprocess the image
        image = Image.open(file.stream).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)

        # Load model (only once)
        current_model = load_model()

        # Perform inference
        with torch.no_grad():
            output = current_model(image_tensor)
            probs = torch.sigmoid(output) # Convert logits to probabilities
            pred_mask = (probs > 0.5).float().squeeze().cpu().numpy()  # Apply threshold      

        # Convert predicted mask to image format
        pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

        # Save image to memory buffer
        buffer = io.BytesIO()
        pred_mask_img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode image as base64 string for JSON response
        encoded_mask = base64.b64encode(buffer.read()).decode("utf-8")
       
        # Return result
        return jsonify({
            "message": "Segmentation completed",
            "predicted_mask": encoded_mask
        })

    except Exception as e:
        # Return error if something goes wrong
        return jsonify({"error": str(e)}), 500

# Run the Flask app   
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)