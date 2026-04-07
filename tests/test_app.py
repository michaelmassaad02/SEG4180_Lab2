from app import app
from PIL import Image
import io


def create_test_image():
    image = Image.new("RGB", (256, 256), color="white")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_predict_success():
    client = app.test_client()
    img = create_test_image()

    response = client.post(
        "/predict",
        data={"image": (img, "test.png")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "message" in data
    assert "building_pixels" in data
    assert "total_pixels" in data
    assert "building_ratio" in data


def test_predict_fail_no_image():
    client = app.test_client()
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data