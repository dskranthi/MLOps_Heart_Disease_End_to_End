import os

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import FEATURES, TARGET
from src.features import build_preprocessor

VALID = {"age": 58, "sex": 1, "cp": 4, "trestbps": 130, "chol": 250, "fbs": 0,
         "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3}


@pytest.fixture(scope="module")
def client(synthetic_clean, tmp_path_factory):
    import joblib
    p = tmp_path_factory.mktemp("m") / "m.joblib"
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))])
    joblib.dump(pipe.fit(synthetic_clean[FEATURES], synthetic_clean[TARGET]), p)
    os.environ["MODEL_PATH"] = str(p)
    import app

    return TestClient(app.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_ok(client):
    r = client.post("/predict", json={"features": VALID})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1) and 0.5 <= body["confidence"] <= 1.0


def test_predict_missing_field_is_422(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_predict_out_of_range_is_422(client):
    assert client.post("/predict", json={"features": {**VALID, "age": 500}}).status_code == 422


def test_metrics_endpoint_counts_requests(client):
    client.post("/predict", json={"features": VALID})
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "api_predictions_total" in text
