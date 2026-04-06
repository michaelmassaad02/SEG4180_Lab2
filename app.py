from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

model = os.getenv("MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
port = int(os.getenv("PORT", 5000))
secret_key = os.getenv("SECRET_KEY", "default")

print("Loaded model:", model)
print("Loaded port:", port)

app.config['SECRET_KEY'] = secret_key

# Load pretrained model for sentiment analysis
sentiment_analysis = pipeline("sentiment-analysis", model= model)

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    result = sentiment_analysis(text)

    return jsonify({"input": text, "prediction": result})

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5000)