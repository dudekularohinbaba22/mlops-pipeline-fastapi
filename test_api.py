import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src import train as trainer


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    trainer.promote_if_better(trainer.train(n_estimators=20))


@pytest.fixture()
def client():
    from src.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(client):
    sample = load_breast_cancer().data[0].tolist()
    r = client.post("/predict", json={"features": sample})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {"malignant", "benign"}
    assert 0 <= body["confidence"] <= 1


def test_predict_wrong_length(client):
    r = client.post("/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    assert client.get("/metrics/").status_code == 200
