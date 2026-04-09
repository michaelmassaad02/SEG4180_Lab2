FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.py .
COPY dataset_utils.py .
COPY evaluate_model.py .
COPY train_model.py .
COPY prepare_dataset.py .
COPY house_segmentation_model.pth .

EXPOSE 5000

CMD ["python", "app.py"]