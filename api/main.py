import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import PatientFeatures, PredictionResponse
from heart_risk.config import METADATA_PATH, MODEL_PATH

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("heart-risk-api")
REQUESTS = Counter("heart_api_requests_total", "API requests", ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "API latency", ["path"])
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions", ["label"])
DISCLAIMER = (
    "Educational screening estimate only; not a diagnosis or substitute for clinical advice."
)


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python scripts/train.py` first.")
    return joblib.load(model_path), json.loads(metadata_path.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.metadata = load_artifacts()
    logger.info("model_loaded name=%s", app.state.metadata["model_name"])
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="MLOps assignment API using the UCI Cleveland heart disease cohort.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(time.perf_counter() - started)
        logger.info("request method=%s path=%s status=%s", request.method, path, status)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "healthy", "model": request.app.state.metadata["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request) -> PredictionResponse:
    try:
        frame = pd.DataFrame([patient.model_dump()])
        probability = float(request.app.state.model.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)
    except (ValueError, TypeError) as exc:
        logger.exception("prediction_failed")
        raise HTTPException(
            status_code=422, detail="Unable to process supplied clinical features"
        ) from exc
    label = "presence" if prediction else "absence"
    PREDICTIONS.labels(label).inc()
    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability=round(probability, 6),
        model_version=request.app.state.metadata["trained_at_utc"],
        disclaimer=DISCLAIMER,
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
