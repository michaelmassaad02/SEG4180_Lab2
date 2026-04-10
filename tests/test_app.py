import io
from PIL import Image
from app import app

def test_home():
    """
    Tests the root (/) endpoint.
    Verifies that the API is running and returns a valid response.
    """
    client = app.test_client()
    response = client.get("/")

    # Check that the request was successful
    assert response.status_code == 200

    # Check that the response contains a message
    data = response.get_json()
    assert "message" in data

def test_predict_fail_no_image():
    """
    Tests the /predict endpoint when no image is provided.
    Ensures proper error handling.
    """
    client = app.test_client()
    
    # Send an empty request
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    # Expect a 400 Bad Request response
    assert response.status_code == 400

    # Check that an error message is returned
    data = response.get_json()
    assert "error" in data

def test_predict_success():
    """
    Tests the /predict endpoint with a valid image.
    Verifies that the model processes the input and returns a prediction.
    """
    client = app.test_client()

    # Create a dummy image (white image) for testing
    image = Image.new("RGB", (256, 256), color="white")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Send POST request with image file
    response = client.post(
        "/predict",
        data={"image": (img_bytes, "test.png")},
        content_type="multipart/form-data"
    )

    # Check that the request was successful
    assert response.status_code == 200

    # Verify that the response contains the predicted mask
    data = response.get_json()
    assert "predicted_mask" in data