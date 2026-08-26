import pandas as pd
from pathlib import Path


def test_health_endpoint(test_client):
    """Test the /health endpoint returns correct status."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "healthy"


def test_predict_endpoint_valid_input(test_client, sample_home_features):
    """Test the /predict endpoint with valid input."""
    response = test_client.post("/predict", json=sample_home_features)
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_price" in response_data
    assert isinstance(response_data["predicted_price"], float)
    assert response_data["predicted_price"] > 0


def test_predict_with_missing_fields(test_client, sample_home_features):
    payload = dict(sample_home_features)
    payload["bathrooms"] = None
    payload["sqft_lot"] = None
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_predict_unknown_zipcode(test_client, sample_home_features):
    payload = dict(sample_home_features)
    payload["zipcode"] = "00000"
    response = test_client.post("/predict", json=payload)
    assert response.status_code == 404


def test_predict_future_unseen_examples(test_client):
    path = Path("data/future_unseen_examples.csv")
    if not path.exists():
        path = Path("src/data/future_unseen_examples.csv")
    examples = pd.read_csv(path, dtype={"zipcode": str})
    cols = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "zipcode",
    ]
    for _, row in examples[cols].head(3).iterrows():
        response = test_client.post("/predict", json=row.to_dict())
        assert response.status_code == 200
        assert response.json()["predicted_price"] > 0


def test_predict_does_not_reload_model(test_client, sample_home_features, monkeypatch):
    import pickle

    calls = []
    original_load = pickle.load

    def tracking_load(*args, **kwargs):
        calls.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr("utils.loader.pickle.load", tracking_load)
    assert test_client.post("/predict", json=sample_home_features).status_code == 200
    assert test_client.post("/predict", json=sample_home_features).status_code == 200
    assert calls == []
