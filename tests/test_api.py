from fastapi.testclient import TestClient

import api.main as api_module

VALID_PATIENT = {
    "age": 54,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 3,
}


def test_health_and_prediction(monkeypatch, dummy_artifacts):
    loader = api_module.load_artifacts
    monkeypatch.setattr(
        api_module, "load_artifacts", lambda: loader(*dummy_artifacts)
    )
    with TestClient(api_module.app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=VALID_PATIENT)
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in [0, 1]
        assert 0 <= body["probability"] <= 1
        assert "diagnosis" in body["disclaimer"]


def test_invalid_patient_is_rejected(monkeypatch, dummy_artifacts):
    loader = api_module.load_artifacts
    monkeypatch.setattr(
        api_module, "load_artifacts", lambda: loader(*dummy_artifacts)
    )
    with TestClient(api_module.app) as client:
        response = client.post("/predict", json={**VALID_PATIENT, "age": 200})
        assert response.status_code == 422
