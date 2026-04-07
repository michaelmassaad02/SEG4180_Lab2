from app import app

def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert "message" in data

def test_predict_fail_no_image():
    client = app.test_client()
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data