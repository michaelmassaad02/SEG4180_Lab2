SEG4180 Lab 2 – Secrets Injection CICD and Segmentation Model Training
Author: Michael Massaad, 300293612
Docker Hub Username: michaelmassaad02

This lab demonstrates how to:



Pretrained Model Link: 
https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english 

The API has one endpoint:

POST /predict

It accepts JSON input in the following format:

{
  "text": "Your text here"
}

It returns a JSON response containing the prediction:

{
  "input": "...",
  "prediction": [
    {
      "label": "POSITIVE or NEGATIVE",
      "score": probability
    }
  ]
}

Prerequisites:
- Docker Desktop installed and running

Docker Hub Repository:

https://hub.docker.com/r/michaelmassaad02/model-service

Image Name:
michaelmassaad02/model-service:latest


How to run container locally:

1. Open a terminal.

2. Pull the Docker image from Docker Hub:

   docker pull michaelmassaad02/model-service:latest

3. Run the container:

   docker run -p 5000:5000 michaelmassaad02/model-service:latest

4. The application will start and listen on:

   http://localhost:5000

How to test:

Using PowerShell (Windows):

Invoke-RestMethod -Method POST http://localhost:5000/predict `
-Headers @{"Content-Type"="application/json"} `
-Body '{"text":"Docker is working!"}'

Using curl (Mac/Linux):

curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"Docker is working!"}'

Files included in submission:
- app.py
- Dockerfile
- requirements.txt
- README.txt
- SEG4180_Lab1_APIRunningLocally.png (Screenshot)
- SEG4180_Lab1_DockerHubRepoWithPushedImage.png (Screenshot)
- SEG4180_Lab1_DockerHubRepoWithPushedImage2.png (Screenshot)
- SEG4180_Lab1_DockerRunningAPI.png (Screenshot)