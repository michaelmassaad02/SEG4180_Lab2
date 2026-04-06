from app import app

def testing_prediction_pass():
    client = app.test_client()
    response = client.post('/predict', json={"text": "I love this product!"})

    assert response.status_code == 200

    data = response.get_json()
    assert "input" in data
    assert "prediction" in data

def testing_prediction_fail():
    client = app.test_client()
    response = client.post('/predict', json={})

    assert response.status_code == 400

    data = response.get_json()
    assert "error" in data