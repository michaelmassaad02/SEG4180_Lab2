SEG4180 Lab 2 – Secrets Injection CICD and Segmentation Model Training
Author: Michael Massaad, 300293612
Docker Hub Username: michaelmassaad02

This lab demonstrates how to:

- Implement secrets injection using environment variables and .env
- Build a CI/CD pipeline using GitHub Actions
- Train a UNet segmentation model for aerial building detection
- Evaluate the model using IoU and Dice score
- Deploy the model as a Flask API using Docker

Dataset Used: 
https://huggingface.co/datasets/keremberke/satellite-building-segmentation 

Model:

A UNet-based segmentation model was trained using PyTorch to detect buildings in aerial images.
The model was trained on resized images (256×256) with masks generated from bounding boxes

The API has two endpoint:

GET /

Returns a message confirming the API is running.

POST / predict

Accepts an image file and returns a predicted segmentation mask.

Request format (form-data):
  Key: image
  Type: File

Response Format:
JSON:
  {
  "message": "Segmentation completed",
  "predicted_mask": "base64_encoded_image_string"
  }


Prerequisites:
- Docker Desktop installed and running
- Python (if running locally without Docker)

Docker Hub Repository:

https://hub.docker.com/r/michaelmassaad02/house-segmentation

Image Name:
michaelmassaad02/house-segmentation:lab2


How to run container locally:

1. Open a terminal.

2. Pull the Docker image from Docker Hub:

   docker pull michaelmassaad02/house-segmentation:lab2

3. Run the container:

   docker run -p 5000:5000 michaelmassaad02/house-segmentation:lab2

4. The application will start and listen on:

   http://localhost:5000

How to test:

Using Postman:

- Method: POST
- URL: http://localhost:5000/predict
- Body -> form-data
- Key: image (file)
- Upload an image and send the request

PowerShell (Windows):
Invoke-RestMethod -Method POST http://localhost:5000/predict `
-Form @{image=Get-Item "test_image.png"}

Using curl (Mac/Linux):
curl -X POST http://localhost:5000/predict \
-F "image=@test_image.png"

Files included in submission:
- app.py
- .github\workflows\ci.yml
- model.py
- .env
- .gitignore
- dataset_utils.py
- prepare_dataset.py
- train_model.py
- evaluate_model.py
- tests\test_app.py
- Dockerfile
- requirements.txt
- house_segmentation_model.pth
- loss_curve.png
- prediction_example_1.png
- prediction_example_2.png
- prediction_example_3.png
- prediction_example_4.png
- prediction_example_5.png
- app_test_postman.png
- CI-CD_Pipeline_Runs.png
- model_evaluation.png
- model_training.png