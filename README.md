# Heart Disease Risk — Assignment 1 MLOps Project

This repository turns the official UCI
Cleveland heart-disease cohort into a reproducible binary classifier and serves the selected
pipeline through a monitored FastAPI service.

## Tools Used
Python 3.11 used, Git, VS Code, MLFlow, Ruff, Uvicorn, Docker, Swagger, Prometheus, Grafana was used.
  
## Reproduce locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/eda_data.py
python scripts/train.py
pytest --cov=src/heart_risk --cov=api
ruff check .
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data @scripts/sample_request.json
```

Start MLflow with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Containers and monitoring

Build only the API:

```bash
docker build -t heart-risk-api:1.0.0 .
docker run --rm -p 8000:8000 heart-risk-api:1.0.0
```

Run the API, Prometheus, and Grafana together:

```bash
docker compose up --build
```

- API/Swagger: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `mlops-demo`)

## Kubernetes (local Minikube)

```bash
minikube start
eval $(minikube docker-env)
docker build -t heart-risk-api:1.0.0 .
kubectl apply -f k8s/
kubectl rollout status deployment/heart-risk-api
minikube service heart-risk-api --url
```

For a public registry, replace `heart-risk-api:1.0.0` in `k8s/deployment.yaml` with your
published image reference. The service is `LoadBalancer`; local Minikube supplies access through
`minikube service`.

## Repository map
api/                    FastAPI schemas, prediction, health and metrics endpoints
artifacts/models/       Selected pipeline, metadata, comparison table
artifacts/plots/        EDA and evaluation evidence
data/                   Downloaded raw and cleaned data
docs/                   Screenshot instructions
k8s/                    Deployment and LoadBalancer Service
monitoring/             Prometheus and provisioned Grafana dashboard
report/                 Final report
scripts/                Data, EDA and training entry points
src/heart_risk/         Reusable data and modelling code
tests/                  Data, model and API tests
.github/workflows/      CI quality, testing, training and image build
